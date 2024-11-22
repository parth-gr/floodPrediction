# Flood Prediction Using Multi-layer Perceptron Regression Model
## Introduction
Flood is one of the most common natural disasters in India. Global climate change and extreme weather conditions cause floods globally. In earlier days, Statistical methods and historical data were the most common methods to predict floods. In the present day, machine learning—a powerful tool that is transforming flood forecasting with enhanced accuracy and predictive capabilities. Artificial Neural Networks (ANNs), Support Vector Machines (SVMs), Random Forests, and Long Short-Term Memory (LSTM) Networks are the most popular machine learning techniques to predict floods. 
## Linear Model
The following are a set of methods intended for regression in which the target value is expected to be a linear combination of the features. In mathematical notation, if $\hat{y}$  is the predicted value.
 
 $\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p$

 Across the module, we designate the vector $w = (w_1,..., w_p)$  as coef_ and $w_0$ as intercept_.

 Linear regression fits a linear model with coefficients $w = (w_1, ..., w_p)$   to minimize the residual sum of squares between the observed targets in the dataset and the targets predicted by the linear approximation. Mathematically it solves a problem of the form:

 $\min_{w} || X w - y||_2^2$

## MLP Model
The multilayer perceptron is known as a feed-forward neural network, which generally consists of several layers of neurons. Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function  $𝑓:𝑅^𝑚 \to 𝑅^0$ by training on a dataset, where m is the number of dimensions for input, and 0 is the number of dimensions for output $X=x_1,x_2......x_m$ and a target $y$; it can learn a non-linear function approximator for either classification or regression. Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation $w_1x_1 +w_2x_2+......w_mx_m$, followed by a non-linear activation function $g(): R^m \to R$.
## MLPRegressor
Class MLPRegressor implements a multi-layer perceptron (MLP) that trains using backpropagation with no activation function in the output layer, which can also be seen as using the identity function as an activation function. Therefore, it uses the square error as the loss function, and the output is a set of continuous values.
MLPRegressor also supports multi-output regression, in which a sample can have more than one target.

## Mathematical Foundations
 Here are a few assumptions which need to be kept in mind when building linear regression models:

There is a linear relationship between input and output variables.
The noise or residual error is well-mannered (normal or Gaussian distribution).

Let’s say there is a numerical response variable, Y, and one or more predictor variables, $X_1, X_2,$ etc. Let’s say, hypothetically speaking, the following represents the relationship between Y and X in the real world.
  
   $Y_i = f(X) + error$
   
If $Y_i$  is the ith observed value and $\bar{Y_1}$  is the ith predicted value, then the ith residual or error value is calculated as the following:

$e_1=Y_1- \bar{Y_i}$
## Regularization and Algorithm
MLPRegressor uses parameter alpha for regularization (L2 regularization) term which helps in avoiding overfitting by penalizing weights with large magnitudes. 

MLP trains using Stochastic Gradient Descent, Adam, or L-BFGS. Stochastic Gradient Descent (SGD) updates parameters using the gradient of the loss function with respect to a parameter that needs adaptation, i.e.
$w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
+ \frac{\partial Loss}{\partial w})$
where 
 $\eta$ is the learning rate which controls the step size in the parameter space search. 
Loss  is the loss function used for the network.

## Resuls and Discusion











