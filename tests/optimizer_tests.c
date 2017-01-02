#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"
#include "../src/layer.h"
#include "../src/network.h"
#include "../src/optimizer.h"

int main(){
    // test backpropagation
    int hiddenSize[] = {2};
    void (*hiddenActivations[])(Matrix*) = {sigmoid};
    Network* network = createNetwork(2, 1, hiddenSize, hiddenActivations, 2, softmax);
    double** data = (double**)malloc(sizeof(double*) * 1);
    data[0] = (double*)malloc(sizeof(double*) * 2);
    data[0][0] = 1;
    data[0][1] = 0;
    Matrix* dataR = createMatrix(1, 2, data);
    network->connections[0]->weights->data[0][0] = .5;
    network->connections[0]->weights->data[0][1] = 1;
    network->connections[0]->weights->data[1][0] = -1.5;
    network->connections[0]->weights->data[1][1] = .25;
    network->connections[1]->weights->data[0][0] = .25;
    network->connections[1]->weights->data[0][1] = 1;
    network->connections[1]->weights->data[1][0] = -.5;
    network->connections[1]->weights->data[1][1] = -1;
    forwardPass(network, dataR);
    assert(network->layers[2]->input->data[0][0] >= .474 && network->layers[2]->input->data[0][0] <= .475);
    assert(network->layers[2]->input->data[0][1] >= .520 && network->layers[2]->input->data[0][1] <= .530);

    double** classes = (double**)malloc(sizeof(double*) * 1);
    classes[0] = (double*)malloc(sizeof(double*) * 2);
    classes[0][0] = 0;
    classes[0][1] = 1;
    Matrix* classR = createMatrix(1, 2, classes);

    batchGradientDescent(network, dataR, classR, 1, 1, 1);
    assert(network->connections[1]->weights->data[0][0] <= -.04 &&  network->connections[1]->weights->data[0][0] >= -.05);
    assert(network->connections[1]->weights->data[0][1] >= 1.29 &&  network->connections[1]->weights->data[0][1] <= 1.30);
    assert(network->connections[1]->weights->data[1][0] <= -.84 &&  network->connections[1]->weights->data[1][0] >= -.85);
    assert(network->connections[1]->weights->data[1][1] <= -.65 &&  network->connections[1]->weights->data[1][1] >= -.66);
    assert(network->connections[0]->weights->data[0][0] >= .58 &&  network->connections[0]->weights->data[0][0] <= .59);
    assert(network->connections[0]->weights->data[0][1] >= .94 &&  network->connections[0]->weights->data[0][1] <= .96);
    assert(network->connections[0]->weights->data[1][0] <= -1.4 &&  network->connections[0]->weights->data[1][0] >= -1.6);
    assert(network->connections[0]->weights->data[1][1] >= .24 &&  network->connections[0]->weights->data[1][1] <= .25);

    destroyMatrix(dataR);
    destroyMatrix(classR);
    destroyNetwork(network);

    // test on XOR ([off on] ordering)
    int i;
    data = (double**)malloc(sizeof(double*) * 4);
    for (i = 0; i < 4; i++){
        data[i] = (double*)malloc(sizeof(double) * 2);
    }
    data[0][0] = 0;
    data[0][1] = 0;
    data[1][0] = 0;
    data[1][1] = 1;
    data[2][0] = 1;
    data[2][1] = 0;
    data[3][0] = 1;
    data[3][1] = 1;
    Matrix* trainingData = createMatrix(4, 2, data);
    classes = (double**)malloc(sizeof(double*) * 4);
    for (i = 0; i < 4; i++){
        classes[i] = (double*)malloc(sizeof(double) * 2);
        classes[i][0] = (data[i][0] == 0 && data[i][1] == 0) || (data[i][0] == 1 && data[i][1] == 1) ? 1 : 0;
        classes[i][1] = classes[i][0] == 1 ? 0 : 1;
    }
    Matrix* trainingClasses = createMatrix(4, 2, classes);

    int hiddenSize2[] = {3};
    void (*hiddenActivations2[])(Matrix*) = {tanH};
    Network* network2 = createNetwork(2, 1, hiddenSize2, hiddenActivations2, 2, softmax);

    printf("Starting accuracy of %f\n", accuracy(network2, trainingData, trainingClasses));
    batchGradientDescent(network2, trainingData, trainingClasses, .1, 1000, 1);
    printf("Final accuracy of %f\n", accuracy(network2, trainingData, trainingClasses));
    
    destroyMatrix(trainingData);
    destroyMatrix(trainingClasses);
    destroyNetwork(network2);

    return 0;
}