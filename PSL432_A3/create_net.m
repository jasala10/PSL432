function net = create_net(n_neurons, rescale, activation)

net.n_neurons = n_neurons;
net.activation = activation;
net.n_layers = size(n_neurons, 1);  % no. of layers including input
[net.W, net.W_, net.W__, net.b, net.b_, net.b__, net.v, net.v_, net.v__, ...
  net.y, net.delta, net.dL_dW, net.dL_db] = deal(cell(net.n_layers, 1));

for l = 2:net.n_layers
  if net.activation == "relu" 
    m = rescale(l)*sqrt(2/n_neurons(l - 1));
    net.W{l} = m*randn(n_neurons(l), n_neurons(l - 1));  % if rescale = 1, this is Kaiming initialization of the weights
  else
    m = rescale(l)*sqrt(3/n_neurons(l - 1));
    net.W{l} = m*2*(rand(n_neurons(l), n_neurons(l - 1)) - 0.5);  % if rescale = 1, this is a simplified variant of Xavier initialization
  end    
  net.W_{l} = zeros(n_neurons(l), n_neurons(l - 1));
  net.W__{l} = zeros(n_neurons(l), n_neurons(l - 1));
  if (l < net.n_layers) && (net.activation == "relu")
    %net.b{l} = zeros(n_neurons(l), 1);  % Kaiming initialization
    net.b{l} = 0.2*rand(n_neurons(l), 1);  % sometimes better than Kaiming
  else
    net.b{l} = zeros(n_neurons(l), 1);  % Xavier initialization
    % net.b{l} = 0.2*rand(n_neurons(l), 1);  % sometimes better than Xavier
  end        
  net.b_{l} = zeros(n_neurons(l), 1);
  net.b__{l} = zeros(n_neurons(l), 1);
end
net.adam_1t = 1;
net.adam_2t = 1;

end