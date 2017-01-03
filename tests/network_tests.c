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
    destroyMatrix(predictM);
    destroyMatrix(actual);

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