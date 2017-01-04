#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"
#include "../src/layer.h"
#include "../src/network.h"

int main(){
    // test network creation
    int hiddenSize[] = {3, 5};
    void (*hiddenActivations[])(Matrix*) = {sigmoid, relu};
    Network* network = createNetwork(5, 2, hiddenSize, hiddenActivations, 4, softmax);
    assert(network->numLayers == 4 && network->numConnections == 3); 
    assert(network->layers[0]->size == 5);
    assert(network->layers[1]->size == 3);
    assert(network->layers[2]->size == 5);
    assert(network->layers[3]->size == 4);

    // test forward pass
    double** example_data = (double**)malloc(sizeof(double*) * 2);
    example_data[0] = (double*)malloc(sizeof(double) * 5);
    example_data[1] = (double*)malloc(sizeof(double) * 5);
    int i, j;
    for (i = 0; i < 5; i++){
        example_data[0][i] = (i + 1.0) / 2;
        example_data[1][i] = (i + 1.5) / 2;
    }
    Matrix* example = createMatrix(2, 5, example_data);
    forwardPass(network, example);

    // test cross-entropy loss
    double** A_data = (double**)malloc(sizeof(double*) * 3);
    for (i = 0; i < 3; i++){
        A_data[i] = (double*)malloc(sizeof(double) * 3);
        for (j = 0; j < 3; j++){
            A_data[i][j] = i + j;
        }
    }
    double** B_data = (double**)malloc(sizeof(double*) * 3);
    for (i = 0; i < 3; i++){
        B_data[i] = (double*)malloc(sizeof(double) * 3);
        for (j = 0; j < 3; j++){
            B_data[i][j] = i + j;
        }
    }
    Matrix* predictM = createMatrix(3, 3, A_data);
    Matrix* actual = createMatrix(3, 3, B_data);
    assert(crossEntropyLoss(NULL, predictM, actual, 0) <= 0.001);

    // test prediction
    double** predict_data = (double**)malloc(sizeof(double*) * 1);
    predict_data[0] = (double*)malloc(sizeof(double) * 5);
    predict_data[0][0] = 0.1;
    predict_data[0][1] = 0.2;
    predict_data[0][2] = 0.7;
    predict_data[0][3] = 0;
    predict_data[0][4] = 0;
    Matrix* predictData = createMatrix(1, 5, predict_data);
    destroyMatrix(network->layers[3]->input);
    network->layers[3]->input = predictData;
    int* prediction = predict(network);
    assert(prediction[0] == 2);
    free(prediction);

    // test serialization
    saveNetwork(network, "network.pkl");
    Network* fromFile = readNetwork("network.pkl");
    assert(network->numLayers == fromFile->numLayers);
    assert(network->numConnections == fromFile->numConnections);
    assert(equals(network->connections[0]->weights, fromFile->connections[0]->weights) == 1);
    assert(equals(network->connections[1]->weights, fromFile->connections[1]->weights) == 1);
    assert(equals(network->connections[2]->weights, fromFile->connections[2]->weights) == 1);
    assert(equals(network->connections[0]->bias, fromFile->connections[0]->bias) == 1);
    assert(equals(network->connections[1]->bias, fromFile->connections[1]->bias) == 1);
    assert(equals(network->connections[2]->bias, fromFile->connections[2]->bias) == 1);

    // test everything but without hidden layers
    Network* networkNoHidden = createNetwork(5, 0, NULL, NULL, 4, softmax);
    forwardPass(networkNoHidden, example);
    saveNetwork(networkNoHidden, "network2.pkl");
    Network* fromFile2 = readNetwork("network2.pkl");
    assert(networkNoHidden->numLayers == fromFile2->numLayers);
    assert(networkNoHidden->numConnections == fromFile2->numConnections);
    assert(equals(networkNoHidden->connections[0]->weights, fromFile2->connections[0]->weights) == 1);
    assert(equals(networkNoHidden->connections[0]->bias, fromFile2->connections[0]->bias) == 1);

    // test destroy
    destroyMatrix(predictM);
    destroyMatrix(actual);
    destroyMatrix(example);
    destroyNetwork(network);
    destroyNetwork(fromFile);
    destroyNetwork(networkNoHidden);
    destroyNetwork(fromFile2);

    return 0;
}