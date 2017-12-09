function g = sigmoidGradient(z)
%SIGMOIDGRADIENT returns the gradient of the sigmoid function
%evaluated at z
%   g = SIGMOIDGRADIENT(z) computes the gradient of the sigmoid function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, you should return
%   the gradient for each element.

g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the gradient of the sigmoid function evaluated at
%               each value of z (z can be a matrix, vector or scalar).

[rows, cols] = size(z);
g = zeros(rows, cols);
for i = 1:rows
    for j = 1:cols
        sg = sigmoid(z(i,j));
        g(i, j) = sg * (1 - sg);
    end
end
acc = (1 / 1 + e);

% =============================================================




end
