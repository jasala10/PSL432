
% PSL432 ASSIGNMENT 1 
% AUTHOR: Jake Lance

clear variables;
clc;
clf;

% Set Parameters
Dt = 0.01;                  % time step in s
dur = 10;                   % duration of run in s
wait = ceil(1/Dt);          % interval between target jumps in time steps
q_min = [-2; 0];        
q_max = [1.5; 2.5];     
psi = [1.88; 0.6; 0.48];    % exact psi   
zeta = 0.75;                % exact zeta 
a_max = 25000;      

% (100, 20, 75)
a1 = -1*81;                % 
a2 = -1*18;                %   coefficients for desired dynamics
a3 = -1*50;                %   
alpha = [a1; a2; a3];


% Initialize System
step = 0;                   
a = [0; 0];                 % initial action
 
q_star = [0; 0];            % target's position
q_star_vel = [0; 0];        % target's velocity
torque_star = [0; 0];       % agent's (estimated) desired torque
q_acc_star = [0; 0];        % agent's (estimated) desired acceleration

q = [0; 0];                 
q_vel = [0; 0];             
q_acc = [0; 0];             
torque = [0; 0]; 
torque_vel = [0; 0];

q_est = [0; 0];
q_vel_est = [0; 0];
q_acc_est = [0; 0];
torque_est = [0; 0];
torque_vel_est = [0; 0];

psi_est = 1*psi;            % agent's estimate of psi
zeta_est = 1*zeta;          % agent's estimate of zeta


M = [psi(1) + 2*psi(2)*cos(q(2)), psi(3)+psi(2)*cos(q(2));  
     psi(3)+psi(2)*cos(q(2)), psi(3)];
GAMMA = psi(2)*sin(q(2))*[-1*q_vel(2), -1*(q_vel(1)+q_vel(2)); 
                            q_vel(1) ,  0];

M_est = [psi_est(1) + 2*psi_est(2)*cos(q_est(2)), psi_est(3)+psi_est(2)*cos(q_est(2));
         psi_est(3)+psi_est(2)*cos(q_est(2)),     psi_est(3)];
GAMMA_est = psi_est(2)*sin(q_est(2))*[-1*q_vel_est(2), -1*(q_vel_est(1)+q_vel_est(2)); 
                                         q_vel_est(1), 0];
 

% Prepare data matrix
DATA = zeros(7, 1 + floor(dur/Dt)); 
i_gp = 1;  % index of graph pts
DATA(:, i_gp) = [0; q_star(1); q_star(2); q(1); q(2); a(1); a(2)]; 


% Run Time Loop
for t = Dt:Dt:dur
    
    % PART 1: Set Target Position and Velocity
    step = step + 1;

    
    % (perform every 100 time steps: step 1, 101, 201, ...)
    if mod(step, wait) == 1         
        
        % choose new position and velocity for target, in bounds:
        q_star = [3.5*rand(1,1)-2 ; 2.5*rand(1,1)];         
        q_star_vel = [7*rand(1,1)-3.5 ; 5*rand(1,1) - 2.5]; 

        % adjust velocity if target will go out of bounds    
        % shoulder:
        if q_star(1) + q_star_vel(1) >= 1.5 | ...
            q_star(1) + q_star_vel(1) <= -2
            q_star_vel(1) = 3.2*rand - 2 - q_star(1); 
        end

        % elbow:
        if q_star(2) + q_star_vel(2) >= 2.5 | ...
           q_star(2) + q_star_vel(2) <= 0
            q_star_vel(2) = 2.2*rand - q_star(2); 
        end
    end
    

    % update q_star (every 1 time step) using valid velocity
    q_star(1) = max(-2, min(q_star(1,1) + q_star_vel(1,1)*Dt, 1.5)); 
    q_star(2) = max(0, min(q_star(2,1)+ q_star_vel(2,1)*Dt, 2.5));


    % STEP 2: Compute Command

    % compute the agent's desired acceleration (Hurwitz Dynamics)
    q_acc_star = alpha(2)*(q_vel_est - q_star_vel) + alpha(1)*(q_est - q_star);

    % estimate the desired torque using the desired acceleration
    torque_star = M_est*q_acc_star + GAMMA_est*q_vel_est;

    % compute action, bounding if (usually not) necessary
    a = alpha(3)*(torque_est - torque_star);
    a = min(a_max, max(a, -a_max));
   
    % update estimated and real torque and torque_vel
    torque = torque + Dt*torque_vel; 
    torque_vel = a - zeta*torque;  

    torque_est = torque_est + Dt*torque_vel_est;  
    torque_vel_est = a - zeta_est*torque_est;  

 
    % update real and estimated dynamics matrices
    M = [psi(1) + 2*psi(2)*cos(q(2)), psi(3)+psi(2)*cos(q(2));
         psi(3)+psi(2)*cos(q(2)), psi(3)];
    GAMMA = psi(2)*sin(q(2))*[-1*q_vel(2), -1*(q_vel(1)+q_vel(2)); 
                            q_vel(1)  , 0];

    M_est = [psi_est(1) + 2*psi_est(2)*cos(q_est(2)), psi_est(3)+psi_est(2)*cos(q_est(2));
         psi_est(3)+psi_est(2)*cos(q_est(2)), psi_est(3)];
    GAMMA_est = psi_est(2)*sin(q_est(2))*[-1*q_vel_est(2), -1*(q_vel_est(1)+q_vel_est(2)); 
                                     q_vel_est(1)  , 0];


    % update real and estimated position, velocity, acceleration 
    % (Euler Integration)

    q_est = q_est + Dt*q_vel_est; 
    q_vel_est = q_vel_est + Dt*q_acc_est;
    q_acc_est = M_est\(torque_est - GAMMA_est*q_vel_est); 

    q = q + Dt*q_vel;
    q_vel = q_vel + Dt*q_acc;
    q_acc = M\(torque - GAMMA*q_vel);


    % Record data for plotting
    i_gp = i_gp + 1;
    DATA(:, i_gp) = [t; q_star(1); q_star(2); q(1); q(2); a(1); a(2)];


end   % for t
DATA = DATA(:, 1:i_gp);


% PLOT
figure(1);
set(gcf, 'Name', 'Two-Joint Arm', 'NumberTitle', 'off');
subplot(2, 1, 1);
ylim(1.05*[min(q_min), max(q_max)])
plot(DATA(1, :), DATA(2, :), 'r:');
hold on;
plot(DATA(1, :), DATA(3, :), 'b:');
plot(DATA(1, :), DATA(4, :), 'r')
plot(DATA(1, :), DATA(5, :), 'b')
ylabel('q')
xlabel('t')

subplot(2, 1, 2);
plot(DATA(1, :), DATA(6, :), 'r');
hold on;
plot(DATA(1, :), DATA(7, :), 'b');
ylabel('action');
xlabel('t');
set(gca, 'TickLength', [0, 0]);
