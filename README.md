<div align="center">
    <img src="docs/image.png"></img>
</div>

<br>

[![Build Status](https://travis-ci.org/100/Cranium.svg?branch=master)](https://travis-ci.org/100/Cranium)
[![MIT License](https://img.shields.io/dub/l/vibe-d.svg)](https://github.com/100/Cranium/blob/master/LICENSE)

## *Cranium* is a portable, header-only, feedforward artificial neural network library written in vanilla C99. 

#### It supports fully-connected networks of arbitrary depth and structure, and should be reasonably fast as it uses a matrix-based approach to calculations. It is particularly suitable for low-resource machines or environments in which additional dependencies cannot be installed.

#### Cranium supports CBLAS integration. Simply uncomment line 7 in ```matrix.h``` to enable the BLAS ```sgemm``` function for fast matrix multiplication.

#### Check out the detailed documentation [here](https://100.github.io/Cranium/) for information on individual structures and functions.

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
* **CBLAS support for fast matrix multiplication**
* **Serializable networks**

<hr>

## Usage
Since Cranium is header-only, simply copy the ```src``` directory into your project, and ```#include "src/cranium.h"``` to begin using it. 

Its only required compiler dependency is from the ```<math.h>``` header, so compile with ```-lm```.

If you are using CBLAS, you will also need to compile with ```-lcblas``` and include, via ```-I```, the path to wherever your particular machine's BLAS implementation is. Common ones include [OpenBLAS](http://www.openblas.net/) and [ATLAS](http://math-atlas.sourceforge.net/).

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

// create training data and target values (data collection not shown)
int rows, features, classes;
float** training;
float** classes;

// create datasets to hold the data
DataSet* trainingData = createDataSet(rows, features, training);
DataSet* trainingClasses = createDataSet(rows, classes, classes);

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
destroyDataSet(trainingData);
destroyDataSet(trainingClasses);

// load previous network from file
Network* previousNet = readNetwork("network");
destroyNetwork(previousNet);
```

<hr>

## Building and Testing

To run tests, look in the ```tests``` folder. 

The ```Makefile``` has commands to run each batch of unit tests, or all of them at once.

<hr>

## Contributing

Feel free to send a pull request if you want to add any features or if you find a bug.

Check the issues tab for some potential things to do.