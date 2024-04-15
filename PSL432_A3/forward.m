function net = forward(net, x)

net.y{1} = x;
if net.activation == "relu"
  for l = 2:net.n_layers - 1
    net.y{l} = max(0, net.W{l}*net.y{l - 1} + net.b{l});
  end
elseif net.activation == "tanh"
  for l = 2:net.n_layers - 1
    net.y{l} = tanh(net.W{l}*net.y{l - 1} + net.b{l});
  end
end  
l = net.n_layers;
net.y{l} = net.W{l}*net.y{l - 1} + net.b{l};  % neurons in final layer are affine

end