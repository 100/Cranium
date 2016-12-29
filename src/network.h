// the overall code for the network in which propagation occurs

#include "layer.c"

typedef struct Network_ {
    int numLayers;
    Layer** layers;
    int numConnections;
    Connection** connections;
} Network;

// constructor to create a network given sizes and functions
// hiddenSizes is an array of sizes, where hiddenSizes[i] is the size
// of the ith hidden layer
// hiddenActivations is an array of activation functions,
// where hiddenActivations[i] is the function of the ith hidden layer
Network* createNetwork(int numFeatures, int numHiddenLayers, int* hiddenSizes, void (**hiddenActivations)(Matrix*), int numClasses, void (*outputActivation)(Matrix*));

// will propagate input through entire network
// result will be stored in input field of last layer
// input should be a row vector representing one example
void forwardPass(Network* network, Matrix* input);

// returns indices corresponding to highest-probability classes for each
// example previously inputted
// assumes final output is in the output layer of network
int* predict(Network* network);

void destroyNetwork(Network* network);