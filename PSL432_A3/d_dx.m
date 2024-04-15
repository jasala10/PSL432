function z_grad = d_dx(net, z)

net.delta{net.n_layers} = z;
if net.activation == "relu"
  for l = net.n_layers - 1:-1:2
    net.delta{l} = (net.W{l + 1}'*net.delta{l + 1}) .* sign(net.y{l});
  end

elseif net.activation == "tanh"
  for l = net.n_layers - 1:-1:2
    net.delta{l} = (net.W{l + 1}'*net.delta{l + 1}) .* (1 - net.y{l}.*net.y{l});
  end
end    

z_grad = net.W{2}'*net.delta{2};  % net.delta{1} = transpose of z'*dy{end}/dy{1}

end

