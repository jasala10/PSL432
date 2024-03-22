function net = j_lance_forward_relog(net, x)
    % Assign input to the first layer
    net.y{1} = x;
    net.v{1} = x;

    % Forward pass through layers 2 and 3 with ReLog activation
    for l = 2:net.n_layers - 1
        net.v{l} = (net.W{l} .* net.C{l})*net.y{l - 1} + net.b{l}; 
        net.y{l} = max(0, sign(net.v{l}).*log(1+abs(net.v{l})));
    end
   
    % Output layer 4 computes its output using an affine transformation
    l = net.n_layers;   
    net.v{l} = (net.W{l} .* net.C{l})*[net.y{l-2}; net.y{l-1}] + net.b{l};
    net.y{l} = net.v{l};

end

