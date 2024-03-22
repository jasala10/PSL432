function net = adam(net, dL_dW, dL_db, eta, adam_1, adam_2)

% Receives precomputed gradients, which override but don't replace
% those stored in the net.
% adam_1 is the adam coefficient beta_1; adam_2 is adam's beta_2.
% Kingma & Ba recommend eta 0.001, adam_1 0.9, adam_2 0.999.



net.adam_1t = net.adam_1t*adam_1;
net.adam_2t = net.adam_2t*adam_2;
correction_1 = 1/(1 - net.adam_1t);
correction_2 = 1/(1 - net.adam_2t);
for l = 2:net.n_layers
  net.W_{l} = adam_1*net.W_{l} + (1 - adam_1)*dL_dW{l};                 % first moment
  net.W__{l} = adam_2*net.W__{l} + (1 - adam_2)*(dL_dW{l}.*dL_dW{l});   % second moment
  denom = sqrt(net.W__{l}*correction_2) + 1e-8;                         % *****
  net.W{l} = net.W{l} - (eta*correction_1) * net.W_{l} ./ denom;        % *****
  net.b_{l} = adam_1*net.b_{l} + (1 - adam_1)*dL_db{l};                 % first moment
  net.b__{l} = adam_2*net.b__{l} + (1 - adam_2)*(dL_db{l}.*dL_db{l});   % second moment
  denom = sqrt(net.b__{l}*correction_2) + 1e-8;                         % *****
  net.b{l} = net.b{l} - (eta*correction_1) * net.b_{l} ./ denom;        % *****
end

end
