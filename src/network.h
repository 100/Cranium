#include "std_includes.h"
#include "matrix.h"
#include "functions.h"
#include "layer.h"

#ifndef NETWORK_H
#define NETWORK_H

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
// input should be a matrix where each row is an input
void forwardPass(Network* network, Matrix* input);

// returns indices corresponding to highest-probability classes for each
// example previously inputted
// assumes final output is in the output layer of network
int* predict(Network* network);

// frees network, its layers, and its connections
void destroyNetwork(Network* network);


/*
    Begin functions.
*/

Network* createNetwork(int numFeatures, int numHiddenLayers, int* hiddenSizes, void (**hiddenActivations)(Matrix*), int numClasses, void (*outputActivation)(Matrix*)){
    assert(numFeatures > 0 && numHiddenLayers >= 0 && numClasses > 0);
    Network* network = (Network*)malloc(sizeof(Network));
    
    network->numLayers = 2 + numHiddenLayers;
    Layer** layers = (Layer**)malloc(sizeof(Layer*) * network->numLayers);
    int i;
    for (i = 0; i < network->numLayers; i++){
        // create input
        if (i == 0){
            layers[i] = createLayer(INPUT, numFeatures, NULL);
        }
        //create output
        else if (i == network->numLayers - 1){
            layers[i] = createLayer(OUTPUT, numClasses, outputActivation);
        }
        // create hidden layer
        else{
            layers[i] = createLayer(HIDDEN, hiddenSizes[i - 1], hiddenActivations[i - 1]);
        }
    }
    network->layers = layers;

    network->numConnections = network->numLayers - 1;
    Connection** connections = (Connection**)malloc(sizeof(Connection*) * network->numConnections);
    for (i = 0; i < network->numConnections; i++){
        connections[i] = createConnection(network->layers[i], network->layers[i + 1]);
        initializeConnection(connections[i]);
    }
    network->connections = connections;

    return network;
}

void forwardPass(Network* network, Matrix* input){
    assert(input->cols == network->layers[0]->input->cols);
    destroyMatrix(network->layers[0]->input);
    network->layers[0]->input = copy(input);
    int i;
    Matrix* tmp,* tmp2;
    for (i = 0; i < network->numConnections; i++){
        tmp = multiply(network->layers[i]->input, network->connections[i]->weights);
        tmp2 = addToEachRow(tmp, network->connections[i]->bias);
        destroyMatrix(network->connections[i]->to->input);
        network->connections[i]->to->input = tmp2;
        destroyMatrix(tmp);
        activateLayer(network->connections[i]->to);
    }
}

int* predict(Network* network){
    int i, j, max;
    Layer* outputLayer = network->layers[network->numLayers - 2];
    int* predictions = (int*)malloc(sizeof(int) * outputLayer->input->rows);
    for (i = 0; i < outputLayer->input->rows; i++){
        max = 0;
        for (j = 1; j < outputLayer->size; j++){
            if (outputLayer->input->data[0][j] > outputLayer->input->data[0][max]){
                max = j;
            }
        }
        predictions[i] = max;
    }
    return predictions;
}

void destroyNetwork(Network* network){
    int i;
    for (i = 0; i < network->numLayers; i++){
        destroyLayer(network->layers[i]);
    }
    for (i = 0; i < network->numConnections; i++){
        destroyConnection(network->connections[i]);
    }
    free(network->layers);
    free(network->connections);
    free(network);
}

#endif