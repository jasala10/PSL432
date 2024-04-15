function y_nse = batch_nse(y_star, y_err)
% Compute the normalized squared error of a network's output

y_nse = max( mean(y_err.*y_err, 2)./(var(y_star', 1)') );

end