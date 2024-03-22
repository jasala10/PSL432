function net = j_lance_create_mapnet()


n_neurons = [784; 784; 784; 10];
net.n_neurons = n_neurons;

field_size = [0, 0; 7, 7; 7, 7];
map_size = [28, 28; 28, 28; 28, 28];

test_m = [0; 49; 49; 1568];
% defining m in terms of number of active upstream neurons leads to
% better performance than something like [0; 784; 784; 1568]


net.activation = "relog";

net.n_layers = size(net.n_neurons, 1);    % 4


% Instantiate cell arrays for 15 variables
[net.W, net.W_, net.W__, net.b, net.b_, net.b__, net.v, net.v_, net.v__, ...
  net.y, net.delta, net.dL_dW, net.dL_db, net.C] = deal(cell(net.n_layers, 1));



for l = 2:net.n_layers % l = 2, 3, 4
    m = sqrt(2/test_m(l));
 
    if l == 4 % (10 x 1568)
        net.W{l} = m*randn(n_neurons(l), n_neurons(l - 1)+n_neurons(l-2));
        net.W_{l} = zeros(n_neurons(l), n_neurons(l - 1)+n_neurons(l-2));
        net.W__{l} = zeros(n_neurons(l), n_neurons(l - 1)+n_neurons(l-2));
    else      % (784 x 784)
        net.W{l} = m*randn(n_neurons(l), n_neurons(l - 1)); 
        net.W_{l} = zeros(n_neurons(l), n_neurons(l - 1));
        net.W__{l} = zeros(n_neurons(l), n_neurons(l - 1));
    end
    
    
    net.b{l} = zeros(n_neurons(l), 1);  % Kaiming initialization

    % define these matrices for adam
    net.b_{l} = zeros(n_neurons(l), 1);
    net.b__{l} = zeros(n_neurons(l), 1);
end


for l = 2:net.n_layers-1 % l = 2, 3
    net.C{l} = zeros(size(net.W{l}));
    O = ones(field_size(l, 1), field_size(l, 2)); % field size with ones
    Z = zeros(map_size(l, 1), map_size(l,2));
    lowest_top = map_size(l, 1) - field_size(l, 1) + 1;
    ratio = (lowest_top - 1)/(map_size(l,1) - 1);
    for i = 1:map_size(l, 1)  % for each row of this layer's map
    top = min(lowest_top, 1 + round((i - 1)*ratio));  % top row of input field in upstream map
        for j = 1:map_size(l, 1)  % for each column of this layer's map (there's an input field defined for each neuron)
          left = min(lowest_top, 1 + round((j - 1)*ratio));  % leftmost column of input field in upstream map
          FIELD = Z;
          FIELD(top:(top + field_size(l, 1) - 1), left:(left + field_size(l, 2) - 1)) = O;  % only this square patch of upstream cells projects to this cell
          field = reshape(FIELD, 1, []);  
          k = (i - 1)*map_size(l, 1) + j;  % number of current cell in this layer's map
          net.C{l}(k, :) = field;
        end
    end
end


l = 4;
net.C{l} = zeros(size(net.W{l}));
for k = 1:10
    field = ones(1, 1568);  
    net.C{l}(k, :) = field;
end 



% define variables to keep track of the first and second moments of the
% gradients
net.adam_1t = 1;
net.adam_2t = 1;

end


