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



h = [];
theta = Theta1;
main_X = X;
%this coverts y to a matrix of 1s and 0s
logical_array = (y == [1:num_labels]);
temp1 = Theta1;
temp2 = Theta2;
capital_delta_1 = zeros(size(Theta1));
capital_delta_2 = zeros(size(Theta2));

for i = 1:2,	
	%add a column of ones
	main_X = [ones(m,1) main_X];
	h = sigmoid(main_X * theta');
	
	main_X = h;
	theta = Theta2;
	
	end 



logarithm = log(h);
second_logarithm = log(1-h);
%to multiply the rows of the matrix independently
derivative = sum(-logical_array .* logarithm,2)	- sum((1-logical_array).*second_logarithm,2);
summed_derivative = sum(derivative);

%change the first column of theta to zeros
temp1(:,1) = 0;
temp2(:,1) = 0;

sum_theta1 = sum(sum(temp1 .* temp1,2));
sum_theta2 = sum(sum(temp2 .* temp2,2));

regularization = lambda/(2*m) * (sum_theta1 + sum_theta2);
J = ((1/m) * summed_derivative)+ regularization;




%
%Backward Propagation

%for i = 1:m,
%	a_1 = X(i,:)';
%	a_1 = [1;a_1]; % 401 * 1
%	
%	z_2 = Theta1 * a_1;%25 x 1
%	a_2 = sigmoid(z_2);
%	a_2 = [1;a_2];% 26 x 1
%	
%	z_3 = Theta2 * a_2;
%	a_3 = sigmoid(z_3);% 10 x 1
%	
%	y_t = y == [1:num_labels];
%	
%	delta_3 = a_3 - y_t(i,:)';% 10 x 1
%	delta_2 = (Theta2' * delta_3); %26 x 1
%	delta_2 = delta_2(2: end).* sigmoidGradient(z_2);
%	
%	
%	capital_delta_1 = capital_delta_1 + delta_2 * a_1'; %26 * 401
%	capital_delta_2 = capital_delta_2 + delta_3 * a_2'; %10 * 26
%end


a_1 = [ones(m,1) X];
z_2 = (Theta1 * a_1')'; %5000 x 25
a_2 = (sigmoid(z_2)); %5000 x 25
a_2 = [ones(m,1) a_2]; %5000 x 26
z_3 = a_2 * Theta2'; % 5000 x 10
a_3 = sigmoid(z_3);

delta_3 = a_3 - logical_array; %5000 x 10
delta_2 = delta_3 * Theta2; % 5000 x 26
delta_2(:,1) = []; %5000 x 25
delta_2 = delta_2 .* sigmoidGradient(z_2);
capital_delta_1 = capital_delta_1 + (a_1' * delta_2)'; % 25 x 401
capital_delta_2 = capital_delta_2 + (a_2' * delta_3)';% 10 x 26


Theta1_grad = (1/m) * capital_delta_1 + (lambda/m) * temp1;
Theta2_grad = (1/m) * capital_delta_2 + (lambda/m) * temp2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
