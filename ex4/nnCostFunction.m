function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));
X = [ones(1, size(X)(1)).' X];

disp(sprintf('init sizes of Theta1_grad %d %d', size(Theta1_grad)));
disp(sprintf('init size of Theta2_grad %d %d', size(Theta2_grad)));
disp(sprintf('size of y is %d %d', size(y)));

y_recoded = [];
if num_labels == 10
    for i = 1:size(y, 1)
        vec = zeros(10, 1);
        if y(i) == 0
            vec(10) = 1;
        else
            vec(y(i)) = 1;
        end
        y_recoded = [y_recoded vec];
    end
else
    for i = 1:size(y, 1)
        vec = zeros(num_labels, 1);
        vec(y(i)) = 1;
        y_recoded = [y_recoded vec];
    end
end


% values for backpropagation
% delta3 = zeros(size(y_recoded));
delta3 = zeros(size(y_recoded'));
zgrad = zeros(m, hidden_layer_size);
a2 = zeros(m, hidden_layer_size);
% a3 = zeros(num_labels, m);
a3 = zeros(m, num_labels);
z2 = zeros(m, hidden_layer_size + 1);
z2(1, :) = ones(size(z2(1, :)));
a2 = [ones(1, m).' a2];
zgrad = ones(size(z2));

for i = 1:m
    ht = Theta1 * X(i,:).';
    z2(i,2:end) = ht;
    zgrad(i, :) = sigmoidGradient(z2(i, :));
    ht = sigmoid(ht);

    a2(i,2:end) = ht;

    ht = [1; ht];
    ht = Theta2 * ht;
    ht = sigmoid(ht);

    for k = 1:num_labels
        disp(sprintf('in loop iter %d %d', i, k));
        hyp_val = ht(k);
        a3(i, k) = hyp_val;
        delta3(i, k) = a3(i, k) - y_recoded(k, i);

        J = J - y_recoded(k, i) * log(hyp_val) - (1 - y_recoded(k, i)) * log(1 - hyp_val);
    end
end

disp('part 2');

reg = 0;
for i = 1:input_layer_size
    for j = 1:hidden_layer_size
        reg = reg + Theta1(:,2:end)(j, i)**2;
    end
end
for i = 1:num_labels
    for j = 1:hidden_layer_size
        reg = reg + Theta2(:,2:end)(i, j)**2;
    end
end

reg = reg / (2 * m) * lambda;

J = J / m;

J = J + reg;

% Compute delta2
delta2 = zeros(m, hidden_layer_size);
disp(size(Theta2'));
disp(sprintf('zgrad dims %d %d', size(zgrad)));
for i = 1:m
    delta2(i, :) = (((Theta2' * (delta3(i, :)')) .* (zgrad(i, :)')))(2:end);
end


% Compute Theta1_grad and Theta2_grad
for i = 1:m
    Theta1_grad = Theta1_grad + delta2(i,:)' * X(i,:);
end
for i = 1:m
    Theta2_grad = Theta2_grad + delta3(i,:)' * a2(i,:);
end

disp(sprintf('theta1 grad size %d %d', size(Theta1_grad)));
disp(sprintf('theta2 grad size %d %d', size(Theta2_grad)));

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;


% Do regularization
Theta1_reg = zeros(size(Theta1_grad))
Theta2_reg = zeros(size(Theta2_grad));
for i = 1:size(Theta1_grad)(1)
    for j = 2:size(Theta1_grad)(2)
        Theta1_reg(i, j) = Theta1(i, j);
    end
end
for i = 1:size(Theta2_grad)(1)
    for j = 2:size(Theta2_grad)(2)
        Theta2_reg(i, j) = Theta2(i, j);
    end
end
Theta1_reg = Theta1_reg * lambda / m;
Theta1_grad = Theta1_grad + Theta1_reg;

Theta2_reg = Theta2_reg * lambda / m;
Theta2_grad = Theta2_grad + Theta2_reg;

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

disp(sprintf('grad size %d', size(grad)));

end
