#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"
#include "../src/layer.h"
#include "../src/network.h"
#include "../src/optimizer.h"

int main(){
    // test backpropagation correctness
    size_t hiddenSize[] = {2};
    void (*hiddenActivations[])(Matrix*) = {sigmoid};
    Network* network = createNetwork(2, 1, hiddenSize, hiddenActivations, 2, softmax);
    float** data = (float**)malloc(sizeof(float*) * 1);
    data[0] = (float*)malloc(sizeof(float*) * 2);
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

    float** classes = (float**)malloc(sizeof(float*) * 1);
    classes[0] = (float*)malloc(sizeof(float*) * 2);
    classes[0][0] = 0;
    classes[0][1] = 1;
    Matrix* classR = createMatrix(1, 2, classes);

    batchGradientDescent(network, dataR, classR, CROSS_ENTROPY_LOSS, dataR->rows, 1, 0, 0, 0, 1, 0, 1);
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
    data = (float**)malloc(sizeof(float*) * 4);
    for (i = 0; i < 4; i++){
        data[i] = (float*)malloc(sizeof(float) * 2);
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
    classes = (float**)malloc(sizeof(float*) * 4);
    for (i = 0; i < 4; i++){
        classes[i] = (float*)malloc(sizeof(float) * 2);
        classes[i][0] = (data[i][0] == 0 && data[i][1] == 0) || (data[i][0] == 1 && data[i][1] == 1) ? 1 : 0;
        classes[i][1] = classes[i][0] == 1 ? 0 : 1;
    }
    Matrix* trainingClasses = createMatrix(4, 2, classes);

    size_t hiddenSize2[] = {3};
    void (*hiddenActivations2[])(Matrix*) = {tanH};
    Network* network2 = createNetwork(2, 1, hiddenSize2, hiddenActivations2, 2, softmax);

    printf("\nTESTING ON XOR:\n");
    printf("Starting accuracy of %f\n", accuracy(network2, trainingData, trainingClasses));
    batchGradientDescent(network2, trainingData, trainingClasses, CROSS_ENTROPY_LOSS, trainingData->rows, .3, 0, .01, .9, 1000, 1, 1);
    printf("Final accuracy of %f\n", accuracy(network2, trainingData, trainingClasses));

    // test on above or below y=x^2 [x y ordering] [below above ordering]
    float** dataF = (float**)malloc(sizeof(float*) * 100);
    for (i = 0; i < 100; i++){
        dataF[i] = (float*)malloc(sizeof(float) * 2);
        dataF[i][0] = 1.0 * (i + 1) * (rand() % 50);
        dataF[i][1] = 1.0 * (i + 1) * (rand() % 50);
    }
    Matrix* trainingDataF = createMatrix(100, 2, dataF);
    float** classesF = (float**)malloc(sizeof(float*) * 100);
    for (i = 0; i < 100; i++){
        classesF[i] = (float*)malloc(sizeof(float) * 2);
        classesF[i][0] = dataF[i][0] * dataF[i][0] <= dataF[i][1] ? 1 : 0;
        classesF[i][1] = classesF[i][0] == 1 ? 0 : 1;
    }
    Matrix* trainingClassesF = createMatrix(100, 2, classesF);

    size_t hiddenSizeF[] = {3};
    void (*hiddenActivationsF[])(Matrix*) = {tanH};
    Network* networkF = createNetwork(2, 1, hiddenSizeF, hiddenActivationsF, 2, softmax);

    printf("\nTESTING ON PARABOLA:\n");
    printf("Starting accuracy of %f\n", accuracy(networkF, trainingDataF, trainingClassesF));
    batchGradientDescent(networkF, trainingDataF, trainingClassesF, CROSS_ENTROPY_LOSS, 20, .01, 0, .01, .5, 1000, 1, 1);
    printf("Final accuracy of %f\n", accuracy(networkF, trainingDataF, trainingClassesF));

    // test on regression on y=x^2 + 15
    float** dataReg = (float**)malloc(sizeof(float*) * 1000);
    for (i = 0; i < 1000; i++){
        dataReg[i] = (float*)malloc(sizeof(float) * 1);
        dataReg[i][0] = rand() % 20;
    }
    Matrix* trainingDataReg = createMatrix(1000, 1, dataReg);
    float** classesReg = (float**)malloc(sizeof(float*) * 1000);
    for (i = 0; i < 1000; i++){
        classesReg[i] = (float*)malloc(sizeof(float) * 1);
        classesReg[i][0] = dataReg[i][0] * dataReg[i][0] + 15;
    }
    Matrix* trainingClassesReg = createMatrix(1000, 1, classesReg);

    size_t hiddenSizeReg[] = {20};
    void (*hiddenActivationsReg[])(Matrix*) = {relu};
    Network* networkReg = createNetwork(1, 1, hiddenSizeReg, hiddenActivationsReg, 1, linear);
    
    printf("\nTESTING ON PARABOLA REGRESSION:\n");
    float** oneEx = (float**)malloc(sizeof(float*));
    oneEx[0] = (float*)malloc(sizeof(float));
    oneEx[0][0] = 5.0;
    Matrix* oneExData = createMatrix(1, 1, oneEx);
    forwardPass(networkReg, oneExData);
    printf("Initially maps 5 to %f but should map to 40\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0][0]);
    batchGradientDescent(networkReg, trainingDataReg, trainingClassesReg, MEAN_SQUARED_ERROR, 20, .01, 0, .001, .9, 500, 1, 1);
    forwardPass(networkReg, oneExData);
    printf("After training maps 5 to %f but should map to 40\n", networkReg->layers[networkReg->numLayers - 1]->input->data[0][0]);

    // test parameter set
    ParameterSet params = {networkReg, trainingDataReg, trainingClassesReg, MEAN_SQUARED_ERROR, 20, .01, 0, .001, .9, 200, 1, 1};
    optimize(params);
    
    destroyMatrix(trainingData);
    destroyMatrix(trainingClasses);
    destroyNetwork(network2);
    destroyMatrix(trainingDataF);
    destroyMatrix(trainingClassesF);
    destroyNetwork(networkF);
    destroyMatrix(oneExData);
    destroyMatrix(trainingDataReg);
    destroyMatrix(trainingClassesReg);
    destroyNetwork(networkReg);

    return 0;
}