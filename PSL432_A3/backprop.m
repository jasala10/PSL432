function net = backprop(net, delta_out)

n_m = size(delta_out, 2);
l = net.n_layers;
net.delta{l} = delta_out;  % transpose of dL/dv
net.dL_dW{l} = net.delta{l}*net.y{l - 1}'/n_m;
net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
if net.activation == "relu"
  for l = net.n_layers - 1:-1:2
    net.delta{l} = (net.W{l + 1}'*net.delta{l + 1}) .* sign(net.y{l});
    net.dL_dW{l} = net.delta{l}*net.y{l - 1}'/n_m;
    net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
  end
elseif net.activation == "tanh"
  for l = net.n_layers - 1:-1:2
    net.delta{l} = (net.W{l + 1}'*net.delta{l + 1}) .* (1 - net.y{l}.*net.y{l});
    net.dL_dW{l} = net.delta{l}*net.y{l - 1}'/n_m;
    net.dL_db{l} = sum(net.delta{l}, 2)/n_m;
  end
end
net.delta{1} = net.W{2}'*net.delta{2};

end
