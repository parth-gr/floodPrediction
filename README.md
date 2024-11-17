# Flood Prediction Using Multi-layer Perceptron Regression Model
## Introduction
Flood is one of the most common natural disasters in India. Global climate change and extreme weather conditions cause floods globally. In earlier days, Statistical methods and historical data were the most common methods to predict floods. In the present day, machine learning—a powerful tool that is transforming flood forecasting with enhanced accuracy and predictive capabilities. Artificial Neural Networks (ANNs), Support Vector Machines (SVMs), Random Forests, and Long Short-Term Memory (LSTM) Networks are the most popular machine learning techniques to predict floods. 

## MLP Model
The multilayer perceptron is known as a feed-forward neural network which  in general will consist of several layers of neurons. Multi-layer Perceptron (MLP) is a supervised learning algorithm that learns a function  $𝑓:𝑅^𝑚 \to 𝑅^0$ by training on a dataset, where m is the number of dimensions for input, and 0 is the number of dimensions for output $X=x_1,x_2......x_m$ and a target $y$; it can learn a non-linear function approximator for either classification or regression. Each neuron in the hidden layer transforms the values from the previous layer with a weighted linear summation $w_1x_1 +w_2x_2+......w_mx_m$, followed by a non-linear activation function $g():R^m \to R$.
## Mathematical Foundations
 Here are a few assumptions which need to be kept in mind when building linear regression models:

There is a linear relationship between input and output variables.
The noise or residual error is well-mannered (normal or Gaussian distribution).

Let’s say there is a numerical response variable, Y, and one or more predictor variables, $X_1, X_2,$ etc. Let’s say, hypothetically speaking, the following represents the relationship between Y and X in the real world.
  
   $Y_i = f(X) + error$
If $Y_i$  is the ith observed value and $\cap{Y_1}$  is the ith predicted value, then the ith residual or error value is calculated as the following:










