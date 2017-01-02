#include "../src/std_includes.h"
#include "../src/matrix.h"
#include "../src/function.h"
#include "../src/layer.h"
#include "../src/network.h"
#include "../src/optimizer.h"

int main(){
    // test on XOR ([off on] ordering)
    int i;
    double** data = (double**)malloc(sizeof(double*) * 4);
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
    double** classes = (double**)malloc(sizeof(double*) * 4);
    for (i = 0; i < 4; i++){
        classes[i] = (double*)malloc(sizeof(double) * 2);
        classes[i][0] = (data[i][0] == 0 && data[i][1] == 0) || (data[i][0] == 1 && data[i][1] == 1) ? 1 : 0;
        classes[i][1] = classes[i][0] == 1 ? 0 : 1;
    }
    Matrix* trainingClasses = createMatrix(4, 2, classes);

    int hiddenSize[] = {2};
    void (*hiddenActivations[])(Matrix*) = {tanH};
    Network* network = createNetwork(2, 1, hiddenSize, hiddenActivations, 2, softmax);

    printf("Starting accuracy of %f\n", accuracy(network, trainingData, trainingClasses));
    batchGradientDescent(network, trainingData, trainingClasses, .1, 10000, 1);
    printf("Final accuracy of %f\n", accuracy(network, trainingData, trainingClasses));
    
    destroyMatrix(trainingData);
    destroyMatrix(trainingClasses);
    destroyNetwork(network);

    return 0;
}