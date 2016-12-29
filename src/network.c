#include "network.h"
#include <stdio.h>

Network* createNetwork(int numFeatures, int numHiddenLayers, int* hiddenSizes, void (**hiddenActivations)(Matrix*), int numClasses, void (*outputActivation)(Matrix*)){
    assert(numFeatures > 0 && numHiddenLayers >= 0 && numClasses > 0);
    Network* network = (Network*)malloc(sizeof(Network));
    
    network->numLayers = 2 + numHiddenLayers;
    Layer** layers = (Layer**)malloc(sizeof(Layer*) * network->numLayers);
    int i;
    for (i = 0; i < network->numLayers; i++){
        if (i == 0){ // create input
            layers[i] = createLayer(INPUT, numFeatures, NULL);
        }
        else if (i == network->numLayers - 1){ //create output
            layers[i] = createLayer(OUTPUT, numClasses, outputActivation);
        }
        else{ // create hidden layer
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
    assert(input->rows == 1 && input->cols == network->layers[0]->input->cols);
    destroyMatrix(network->layers[0]->input);
    network->layers[0]->input = copy(input);
    int i;
    Matrix* tmp,* tmp2;
    for (i = 0; i < network->numConnections; i++){
        tmp = multiply(network->layers[i]->input, network->connections[i]->weights);
        tmp2 = add(tmp, network->connections[i]->bias);
        destroyMatrix(network->connections[i]->to->input);
        network->connections[i]->to->input = tmp2;
        destroyMatrix(tmp);
        activateLayer(network->connections[i]->to);
    }
}

int predict(Network* network){
    int i;
    int max = 0;
    Layer* outputLayer = network->layers[network->numLayers - 2];
    for (i = 1; i < outputLayer->size; i++){
        if (outputLayer->input->data[0][i] > outputLayer->input->data[0][max]){
            max = i;
        }
    }
    return max;
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