#include "../src/network.c"
#include <stdio.h>

int main(){
    // test network creation
    int hiddenSize = 3;
    void (*hiddenActivations)(Matrix*) = sigmoid;
    Network* network = createNetwork(5, 1, &hiddenSize, &hiddenActivations, 4, softmax);
    assert(network->numLayers == 3 && network->numConnections == 2); 
    assert(network->layers[0]->size == 5);
    assert(network->layers[1]->size == 3);
    assert(network->layers[2]->size == 4);

    // test forward pass
    double** example_data = (double**)malloc(sizeof(double*) * 2);
    example_data[0] = (double*)malloc(sizeof(double) * 5);
    example_data[1] = (double*)malloc(sizeof(double) * 5);
    int i;
    for (i = 0; i < 5; i++){
        example_data[0][i] = (i + 1.0) / 2;
        example_data[1][i] = (i + 1.5) / 2;
    }
    Matrix* example = createMatrix(2, 5, example_data);
    forwardPass(network, example);

    // test prediction
    int* prediction = predict(network);
    assert(prediction[0] >= 0 && prediction[0] <= 3);
    assert(prediction[1] >= 0 && prediction[1] <= 3);
    free(prediction);

    // test destroy
    destroyMatrix(example);
    destroyNetwork(network);

    return 0;
}