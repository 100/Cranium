<div align="center">
    <img src="image.png"></img>
</div>

<br>

<div align="center">
    <img src=""></img>
</div>

## **Cranium** is a portable, header-only, feed-forward artificial neural network framework written in vanilla C99. 

#### It supports fully-connected networks of arbitrary depth and structure, and should be reasonably fast as it uses a matrix-based approach to calculations. It is particularly suitable for low-resource machines or environments in which additional dependencies cannot be installed.

<hr>

## Features
* **Activation functions**
    * sigmoid
    * ReLU
    * tanh
    * softmax (*classification*)
    * linear (*regression*)
* **Loss functions**
    * Cross-entropy loss (*classification*)
    * Mean squared error (*regression*)
* **Optimization algorithms** 
    * Batch Gradient Descent
    * Stochastic Gradient Descent
    * Mini-Batch Stochastic Gradient Descent
* **L2 Regularization**
* **Learning rate annealing**
* **Simple momentum**
* **Fan-in weight initialization**
* **Serializable network**

<hr>

## Usage
Since **Cranium** is header-only, simply copy the ```src``` directory into your project, and ```#include "src/cranium.h"``` to begin using it. 

Its only compiler dependency is from the ```<math.h>``` header, so compile with ```-lm```.

It has been tested to work perfectly fine with any level of gcc optimization, so feel free to use them. 

<hr>

## Example

```c
#include "cranium.h"

/*
This basic example program is the skeleton of a classification problem.
The training data should be in matrix form, where each row is a data point, and
    each column is a feature. 
The training classes should be in matrix form, where the ith row corresponds to
    the ith training example, and each column is a 1 if it is of that class, and
    0 otherwise. Each example may only be of 1 class.
*/

// create training data and target values
int rows, features, classes;
double** training;
double** classes;

// create matrices to hold the data
Matrix* trainingData = createMatrix(rows, features, training);
Matrix* trainingClasses = createMatrix(rows, classes, classes);

// create network with 2 input neurons, 1 hidden layer with sigmoid
// activation function and 5 neurons, and 2 output neurons with softmax 
// activation function
srand(time(NULL));
size_t hiddenSize[] = {5};
Activation hiddenActivation[] = {sigmoid};
Network* net = createNetwork(2, 1, hiddenSize, hiddenActivation, 2, softmax);

// train network with cross-entropy loss using Mini-Batch SGD
ParameterSet params;
params.network = net;
params.data = trainingData;
params.classes = trainingClasses;
params.lossFunction = CROSS_ENTROPY_LOSS;
params.batchSize = 20;
params.learningRate = .01;
params.searchTime = 5000;
params.regularizationStrength = .001;
params.momentumFactor = .9;
params.maxIters = 10000;
params.shuffle = 1;
params.verbose = 1;
optimize(params);

// test accuracy of network after training
printf("Accuracy is %f\n", accuracy(net, trainingData, trainingClasses));

// get network's predictions on input data after training
forwardPass(net, trainingData);
int* predictions = predict(net);
free(predictions);

// save network to a file
saveNetwork(net, "network");

// free network and data
destroyNetwork(net);
destroyMatrix(trainingData);
destroyMatrix(trainingClasses);

// load previous network from file
Network* previousNet = readNetwork("network");
```

<hr>

## Building and Testing

To run tests, look in the ```tests``` folder. 

The ```Makefile``` has commands to run each batch of unit tests, or all of them at once.