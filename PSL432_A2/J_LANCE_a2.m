% Every breath we take, every step we make, can be filled 
% with peace, joy and serenity.

clear variables

% Load the MNIST data
load 'MNIST.mat';


% Create the network and define simulation variables
y = j_lance_create_mapnet();        % network 

n_batches = 600;            % minibatches per epoch
n_m = 100;                  % examples per minibatch 
n_m_final = 10000;          % examples in the final testing minibatch
eta = 0.001;                % learning rate

test_incorrect = zeros(10, 1);  % number incorrect per epoch of testing


for epoch = 1:10
    
    % BEGIN TRAINING

    % start of epoch, so shuffle the data
    shuffle = randperm(size(TRAIN_images, 1));
    TRAIN_images = TRAIN_images(shuffle, :);
    TRAIN_answers = TRAIN_answers(shuffle, :);
    TRAIN_labels = TRAIN_labels(shuffle, :);

    incorrect_guesses_epoch = 0;

    disp(["a new epoch has begun"]) 

    for batch = 1:n_batches 
        % extract minibatch
        start_idx = (batch - 1) * n_m + 1;
        end_idx = start_idx + n_m - 1;
        minibatch_images = TRAIN_images(start_idx:end_idx, :);
        minibatch_labels = TRAIN_labels(start_idx:end_idx, :);
    
        % forward pass
        y = j_lance_forward_relog(y, minibatch_images');
        
        output = y.y{4};
        soft_output = exp(output) ./ sum(exp(output), 1);

        % DOUBLE CHECK: calculate error and check for incorrect guesses
        [~, predicted_labels] = max(soft_output);
        [~, true_labels] = max(minibatch_labels');
        incorrect(epoch) = sum(predicted_labels ~= true_labels);


        % calculate error and loss
        e = soft_output - TRAIN_labels(start_idx:end_idx, :)';  
        L = 0.5*(e*e');

        % backward pass
        y = j_lance_backprop_relog(y, e);      

        % adam
        y = adam(y, y.dL_dW, y.dL_db, eta, 0.9, 0.999);
    
    end % TRAINING
    
    % BEGIN TESTING

    % we run the network on the entire test set, in one "minibatch" 
    % of 10,000 test examples 

    shuffle = randperm(size(TEST_images, 1));
    TEST_images = TEST_images(shuffle, :);
    TEST_answers = TEST_answers(shuffle, :);
    TEST_labels = TEST_labels(shuffle, :);
    
    start_idx = 1;
    end_idx = 10000;
    minibatch_images = TEST_images(start_idx:end_idx, :);
    minibatch_answers = TEST_answers(start_idx:end_idx, :);
    minibatch_labels = TEST_labels(start_idx:end_idx, :);

    % forward pass
    y = j_lance_forward_relog(y, minibatch_images');

    output = y.y{4};
    soft_output = exp(output) ./ sum(exp(output), 1);

    % check for incorrect guesses
    [~, predicted_labels] = max(soft_output);
    [~, true_labels] = max(minibatch_labels');
    test_incorrect(epoch) = sum(predicted_labels ~= true_labels);
    
end

% disp([incorrect; "incorrect"])
disp([test_incorrect; "test_incorrect"]) 

