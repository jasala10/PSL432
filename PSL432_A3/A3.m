% This is my final copy!

clc
clear variables

% these are used to pass non-input information to f and test_policy
% functions:

global n_q psi zeta q_min q_max n_steps Dt r

psi = [1.88; 0.6; 0.48];
zeta = 0.75;
q_max = [1.5; 2.5];
q_min = [-2; 0];
n_steps = 20;


% Set up state dynamics
n_q = 2;
n_s = 3*n_q;
n_a = n_q;

s_star = [0.5; 0.5; 0; 0; 0; 0];

% Define task
r = @(s, a) -10*(s - s_star)'*(s - s_star) - 0.01*a'*a;
dur = 20*Dt;
Dt = 0.1;
T = floor(1+ dur);

% Set up learning

    % Hyperparamters
    eta_mu = 3e-8;
    eta_f = 1e-4;
    eta_r = 1e-2;
    eta_V = 1e-4;
    tau = 3e-4;
    a_sd = 0.1;
    adam_1 = 0.9;           % Used for both ADAM calls
    adam_2 = 0.999;         % Used for both ADAM calls

    % Buffer
    n_rollouts = 1000;
    n_m = 100;
    n_buf = 1000000;
    buf_fill = 0;
    buf_i = 0;
    BUFFER = zeros(2*n_s + n_a + 1, n_buf);


% Set test
s_test = [-1; 1; 0; 0; 0; 0];


% Seed random-number generator
rng(4);


% Create nets
mu = create_net([6; 250; 250; 2], [0; 0.5; 0.5; 0.5], "relu");
f_est = create_net([8; 100; 100; 6], [0; 0.5; 0.5; 0.5], "relu");
r_est = create_net([8; 50; 50; 1], [0; 0.5; 0.5; 0.5], "relu");
V_est = create_net([6; 100; 100; 1], [0; 0.5; 0.5; 0.5], "relu");
f_tgt = create_net([8; 100; 100; 6], [0; 0.5; 0.5; 0.5], "relu");
V_tgt = create_net([6; 100; 100; 1], [0; 0.5; 0.5; 0.5], "relu");


% Assess initial policy
G = test_policy(mu, s_test);
fprintf('At rollout 0, G = %.3f\n', G);


% TRAIN POLICY
for rollout = 1:n_rollouts


    % choose a sensible initial state
    q1_init = q_min(1) + (q_max(1) - q_min(1)) * rand();
    q2_init = q_min(2) + (q_max(2) - q_min(2)) * rand();
    other = rand(n_s - 2, 1) - 0.5;


    s = [q1_init; q2_init; other];
    for t = 1:Dt:dur


        % Compute transition
        mu = forward(mu, s);
        a = mu.y{end} + a_sd*randn(n_a, 1);
        s_next = f(s, a);

        % Store in the buffer
        BUFFER(:, buf_i +1) = [s; a; r(s, a); s_next];
        buf_i = mod(buf_i + 1, n_buf);
        buf_fill = min(buf_fill + 1, n_buf);

        % Choose a minibatch from buffer
        i = ceil(buf_fill*rand(1, n_m));
        s_ = BUFFER(1:n_s, i);
        a_ = BUFFER(n_s + 1:n_s + n_a, i);
        r_ = BUFFER(n_s + n_a + 1:n_s + n_a +1, i);
        s_next_ = BUFFER(n_s + n_a + 2:end, i);


        % ADJUST CRITIC

            % train <r>

            r_est = forward(r_est, [s_; a_]);
            r_e = r_est.y{end} - r_;
            r_L = 0.5 * (r_e*r_e');
            r_est = backprop(r_est, r_e);
            r_est = adam(r_est, r_est.dL_dW, r_est.dL_db, eta_r, adam_1, adam_2);

           % train <f>

            f_est = forward(f_est, [s_; a_]);
            V_est = forward(V_est, f_est.y{end});
            e1 = V_est.y{end};

            V_est = forward(V_est, s_next_);
            e2 = V_est.y{end};

            f_e = e1 - e2;
            f_L = 0.5 * (f_e*f_e');

            dL_de = f_e;
            de_df = d_dx(V_est, f_e);
            dL_df = d_dx(f_est, de_df);

            f_est = backprop(f_est, dL_df(1:6, :));
            f_est = adam(f_est, f_est.dL_dW, f_est.dL_db, eta_f, adam_1, adam_2);

            % train <V>

            % V(s)
            V_est = forward(V_est, s_); % return starting at s_, using policy

            % r(s)
            mu = forward(mu, s_);
            a_ = mu.y{end} + a_sd*randn(n_a, 1);
            r_est = forward(r_est, [s_; a_]);

            % V(s+1) = V(f(s, mu(s)))
            % method 1:
            % f_est = forward(f_est, [s_; a_]); % you might want to use s_next_ instead.
            % mu = forward(mu, f_est.y{end});
            % a_next_ = mu.y{end} + a_sd*randn(n_a, 1);
            % f_tgt = forward(f_tgt, [f_est.y{end}; a_next_]); % maybe [s_, a_next];
            % V_tgt = forward(V_tgt, f_tgt.y{end});

            % % % method 2:  (V_nse to zero! better!!)
            mu = forward(mu, s_next_);
            a_next_ = mu.y{end};
            f_tgt = forward(f_tgt, [s_next_; a_next_]);
            V_tgt = forward(V_tgt, f_tgt.y{end});

            V_e = V_est.y{end} - r_est.y{end}*Dt - V_tgt.y{end};
            V_L = 0.5*(V_e*V_e');

            % backprop accordingly
            V_est = backprop(V_est, V_e);
            V_est = adam(V_est, V_est.dL_dW, V_est.dL_db, eta_V, adam_1, adam_2);

        % ADJUST ACTOR

            mu = forward(mu, s_);
            a_ = mu.y{end};

            % <r>
            r_est = forward(r_est, [s_; a_]);
            dr = d_dx(r_est, ones(1, n_m));
            dr_da = dr(n_s+1:end, :);

            % <f>
            f_est = forward(f_est, [s_; a_]);
            df = d_dx(f_est, ones(n_s, n_m));
            df_da = df(n_s + 1:end, :);

            % <V>
            f_est = forward(f_est, [s_; a_]);
            V_est = forward(V_est, f_est.y{end});
            dV_df = d_dx(V_est, ones(1, n_m));
            dV_df_df_dx = d_dx(f_est, dV_df);
            dV_da = dV_df_df_dx(n_s+1:end, :);



            %f_est = d_dx(f_est, dV_df);

            % dV_df = dV_df(n_s + 1:end, :); % Extract only the part corresponding to f

            % dV_da = dV_df*df_da + dr_da;


            % f_est = forward(f_est, [s_; a_]);
            % V_est = forward(V_est, f_est.y{end});
            % dV_df = d_dx(V_est, ones(1, n_m));
            %
            % % we have df still
            %
            % dV_da = dV_df*df_da + dr_da;
            mu = backprop(mu, -(dV_da+dr_da*Dt));
            mu = adam(mu, mu.dL_dW, mu.dL_db, eta_mu, adam_1, adam_2);



        % Nudge target net towards learning one
        for l = 2:f_est.n_layers
            f_tgt.W{l} = f_tgt.W{l} + tau*(f_est.W{l} - f_tgt.W{l});
            f_tgt.b{l} = f_tgt.b{l} + tau*(f_est.b{l} - f_tgt.b{l});
        end
        for l = 2:V_est.n_layers
            V_tgt.W{l} = V_tgt.W{l} + tau*(V_est.W{l} - V_tgt.W{l});
            V_tgt.b{l} = V_tgt.b{l} + tau*(V_est.b{l} - V_tgt.b{l});
        end

        % Update s
        s = f(s,a);

    end % for t

    % Test policy
    if mod(rollout, 100) == 0
        G = test_policy(mu, s_test);
        r_nse = batch_nse(r_, r_est.y{end} - r_);
        f_nse = batch_nse(s_next_, f_e);
        V_nse = batch_nse(r_est.y{end} + V_est.y{end}, V_e);

        fprintf('At rollout %d, G = %.3f, r_nse = %.4f, f_nse = %.4f, V_nse = %.4f\n', rollout, G, r_nse, f_nse, V_nse);
    end


end % for rollout

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% State dynamics function: f(s,a)

function s_next = f(s,a)
    % f should handle single examples of s and a, not minibatches
    % f should prevent the joints from going outside of their motion ranges
    % it should euler-update q before q_vel and q_acc

    global psi zeta q_min q_max Dt


    q = s(1:2);
    q_vel = s(3:4);
    q_acc = s(5:6);
    q_jerk = [0; 0]; % initialize

    M = [psi(1) + 2 * psi(2) * cos(q(2)), psi(3) + psi(2) * cos(q(2));
         psi(3) + psi(2) * cos(q(2)), psi(3)];
    GAMMA = psi(2) * sin(q(2)) * [-1 * q_vel(2), -1 * (q_vel(1) + q_vel(2));
                                        q_vel(1), 0];

    torque = M*q_acc + GAMMA*q_vel;
    torque_vel = a - zeta*torque;


    G_der = psi(2)*cos(q(2))*q_vel(2)*[-q_vel(2), -(q_vel(1)+q_vel(2)); q_vel(1), 0]*q_vel + ...
        psi(2)*sin(q(2))*([-q_acc(2), -(q_acc(1)+q_acc(2)); q_acc(1), 0]*q_vel + ...
        [-q_vel(2), -(q_vel(1)+q_vel(2)); q_vel(1), 0]*q_acc);
        % derivative of term with gamma in it

    M_der = [-2*psi(2)*sin(q(2))*q_vel(2), -psi(2)*sin(q(2))*q_vel(2);
       -psi(2)*sin(q(2))*q_vel(2), 0]*q_acc; % first part of M_der

    q_jerk = M\(torque_vel - G_der - M_der);


    q = q + q_vel*Dt;
    q_vel = q_vel + q_acc*Dt;
    q_acc = q_acc + Dt*q_jerk;


    q = max(q_min, min(q_max, q));
    q_vel = q_vel - q_vel.*( (q == q_max).*(q_vel > 0) + ...
                           (q == q_min).*(q_vel < 0) );

    % Construct next state
    s_next = [q; q_vel; q_acc];


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% test_policy function (from DDPG.m)

function G = test_policy(mu, s)
    % modify the function so that it calls my f in the appropriate place

    global n_steps Dt r

    DATA = zeros(4, n_steps);
    G = 0;
    for t = 1:n_steps
      mu = forward(mu, s);  % set noise to 0 for test
      a = mu.y{end};
      G = G + r(s, a)*Dt;
      DATA(:, t) = [t; s(1); a];
      s = f(s, a);
    end  % for t

    % Plot q & a for one test movement
    subplot(2, 1, 1);
    plot(DATA(1, :), DATA(2, :));
    ylim([-1, 1]);
    grid on;
    ylabel('q');
    set(gca, 'TickLength', [0, 0])
    subplot(2, 1, 2);
    plot(DATA(1, :), DATA(3, :));
    ylim(10*[-1, 1]);
    grid on;
    ylabel('action');
    xlabel('t');
    set(gca, 'TickLength', [0, 0]);
    drawnow;

end