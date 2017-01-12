#include "std_includes.h"
#include "matrix.h"
#include "function.h"
#include "layer.h"
#include "network.h"

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

// types of loss functions available to optimize under
typedef enum LOSS_FUNCTION_ {
    CROSS_ENTROPY_LOSS, 
    MEAN_SQUARED_ERROR
} LOSS_FUNCTION;

// convenience struct for easier parameter filling
typedef struct ParameterSet_ {
    Network* network;
    DataSet* data;
    DataSet* classes;
    LOSS_FUNCTION lossFunction;
    size_t batchSize;
    float learningRate;
    float searchTime;
    float regularizationStrength;
    float momentumFactor;
    int maxIters;
    int shuffle;
    int verbose;
} ParameterSet;

// batch gradient descent main function
// $network is the network to be trained
// $data is the training data
// $classes are the true values for each data point in $data, in order
// $batchSize is the size of each batch to use (1 for SGD, #rows for batch)
// $learningRate is the initial learning rate
// $searchTime is the other parameter in the search-and-converge method
// $regularizationStrength is the multiplier for L2 regularization
// $momentumFactor is the multiplier for the moment factor
// $maxIters is the number of epochs to run the algorithm for
// $shuffle, if non-zero, will shuffle the data between iterations
// $verbose, if non-zero, will print loss every 100 epochs
static void batchGradientDescent(Network* network, DataSet* data, DataSet* classes, LOSS_FUNCTION lossFunction, size_t batchSize, float learningRate, float searchTime, float regularizationStrength, float momentumFactor, int maxIters, int shuffle, int verbose);

// optimizes given parameters
static void optimize(ParameterSet params){
    batchGradientDescent(params.network, params.data, params.classes, params.lossFunction, params.batchSize, params.learningRate, params.searchTime, params.regularizationStrength, params.momentumFactor, params.maxIters, params.shuffle, params.verbose);
}


/*
    Begin functions.
*/

void batchGradientDescent(Network* network, DataSet* data, DataSet* classes, LOSS_FUNCTION lossFunction, size_t batchSize, float learningRate, float searchTime, float regularizationStrength, float momentumFactor, int maxIters, int shuffle,  int verbose){
    assert(network->layers[0]->size == data->cols);
    assert(data->rows == classes->rows);
    assert(network->layers[network->numLayers - 1]->size == classes->cols);
    assert(batchSize <= data->rows);
    assert(maxIters >= 1);

    int i, j, k;

    // these will be reused per training instance
    Matrix* errori[network->numLayers];
    Matrix* dWi[network->numConnections];
    Matrix* dbi[network->numConnections];
    Matrix* regi[network->numConnections];
    Matrix* beforeOutputT = createMatrixZeroes(network->layers[network->numLayers - 2]->size, 1);
    for (i = 0; i < network->numConnections; i++){
        errori[i] = createMatrixZeroes(1, network->layers[i]->size);
        dWi[i] = createMatrixZeroes(network->connections[i]->weights->rows, network->connections[i]->weights->cols);
        dbi[i] = createMatrixZeroes(1, network->connections[i]->bias->cols);
        regi[i] = createMatrixZeroes(network->connections[i]->weights->rows, network->connections[i]->weights->cols);
    }
    errori[i] = createMatrixZeroes(1, network->layers[i]->size);

    // these will be reused per training instance if network has hidden layers
    int numHidden = network->numLayers - 2;
    Matrix** WTi,** errorLastTi,** fprimei,** inputTi;
    if (numHidden > 0){
        WTi = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        errorLastTi = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        fprimei = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        inputTi = (Matrix**)malloc(sizeof(Matrix*) * numHidden);
        for (k = 0; k < numHidden; k++){
            WTi[k] = createMatrixZeroes(network->connections[k + 1]->weights->cols, network->connections[k + 1]->weights->rows);
            errorLastTi[k] = createMatrixZeroes(1, WTi[k]->cols);
            fprimei[k] = createMatrixZeroes(1, network->connections[k]->to->size);
            inputTi[k] = createMatrixZeroes(network->connections[k]->from->size, 1);
        }
    }

    // these will be reused per epoch
    Matrix* dWi_avg[network->numConnections];
    Matrix* dbi_avg[network->numConnections];
    Matrix* dWi_last[network->numConnections];
    Matrix* dbi_last[network->numConnections];
    for (i = 0; i < network->numConnections; i++){
        dWi_avg[i] = createMatrixZeroes(network->connections[i]->weights->rows, network->connections[i]->weights->cols);
        dbi_avg[i] = createMatrixZeroes(1, network->connections[i]->bias->cols);
        dWi_last[i] = createMatrixZeroes(network->connections[i]->weights->rows, network->connections[i]->weights->cols);
        dbi_last[i] = createMatrixZeroes(1, network->connections[i]->bias->cols);
    }

    int numBatches = (data->rows / batchSize) + (data->rows % batchSize != 0 ? 1 : 0);
    int training, batch, epoch, layer;
    DataSet** dataBatches,** classBatches;
    epoch = 1;
    while (epoch <= maxIters){
        // shuffle all data and classes but maintain training/class alignment
        if (shuffle != 0){
            shuffleTogether(data, classes);
        }

        // split into overall batches
        dataBatches = createBatches(data, numBatches);
        classBatches = createBatches(classes, numBatches);
        for (batch = 0; batch < numBatches && epoch <= maxIters; batch++, epoch++){
            // find current batch
            int curBatchSize = batch == numBatches - 1 ? (data->rows % batchSize != 0 ? data->rows % batchSize : batchSize) : batchSize;
            DataSet* batchTraining = dataBatches[batch];
            DataSet* batchClasses = classBatches[batch];
            Matrix** splitTraining = splitRows(batchTraining);
            Matrix** splitClasses = splitRows(batchClasses);
            for (training = 0; training < curBatchSize; training++){
                // current data point to train on
                Matrix* example = splitTraining[training];
                Matrix* target = splitClasses[training];
                
                // pass error forward
                forwardPass(network, example);
                
                // calculate each iteration of backpropagation
                for (layer = network->numLayers - 1; layer > 0; layer--){
                    Layer* to = network->layers[layer];
                    Connection* con = network->connections[layer - 1];
                    if (layer == network->numLayers - 1){
                        // calculate output layer's error
                        copyValuesInto(to->input, errori[layer]);
                        if (lossFunction == CROSS_ENTROPY_LOSS){
                            for (j = 0; j < errori[layer]->cols; j++){
                                errori[layer]->data[j] -= target->data[j];
                            }
                        }
                        else{
                            for (j = 0; j < errori[layer]->cols; j++){
                                errori[layer]->data[j] -= target->data[j];
                            }
                        }

                        // calculate dWi and dbi
                        transposeInto(con->from->input, beforeOutputT);
                        multiplyInto(beforeOutputT, errori[layer], dWi[layer - 1]);
                        copyValuesInto(errori[layer], dbi[layer - 1]);
                    }
                    else{
                        // calculate error term for hidden layer
                        int hiddenLayer = layer - 1;
                        transposeInto(network->connections[layer]->weights, WTi[hiddenLayer]);
                        multiplyInto(errori[layer + 1], WTi[hiddenLayer], errorLastTi[hiddenLayer]);
                        copyValuesInto(con->to->input, fprimei[hiddenLayer]);
                        float (*derivative)(float) = activationDerivative(con->to->activation);
                        for (j = 0; j < fprimei[hiddenLayer]->cols; j++){
                            fprimei[hiddenLayer]->data[j] = derivative(fprimei[hiddenLayer]->data[j]);
                        }
                        hadamardInto(errorLastTi[hiddenLayer], fprimei[hiddenLayer], errori[layer]);

                        // calculate dWi and dbi
                        transposeInto(con->from->input, inputTi[hiddenLayer]);
                        multiplyInto(inputTi[hiddenLayer], errori[layer], dWi[layer - 1]);
                        copyValuesInto(errori[layer], dbi[layer - 1]);
                    }
                }

                // add one example's contribution to total gradient
                for (i = 0; i < network->numConnections; i++){
                    addTo(dWi[i], dWi_avg[i]);
                    addTo(dbi[i], dbi_avg[i]);
                }

                // zero out reusable matrices
                zeroMatrix(beforeOutputT);
                for (i = 0; i < network->numConnections; i++){
                    zeroMatrix(errori[i]);
                    zeroMatrix(dWi[i]);
                    zeroMatrix(dbi[i]);
                }
                zeroMatrix(errori[i]);
                if (numHidden > 0){
                    for (i = 0; i < numHidden; i++){
                        zeroMatrix(WTi[i]);
                        zeroMatrix(errorLastTi[i]);
                        zeroMatrix(fprimei[i]);
                        zeroMatrix(inputTi[i]);
                    }
                }
            }

            // calculate learning rate for this epoch
            float currentLearningRate = searchTime == 0 ? learningRate : learningRate / (1 + (epoch / searchTime));
            
            // average out gradients and add learning rate
            for (i = 0; i < network->numConnections; i++){
                scalarMultiply(dWi_avg[i], currentLearningRate / data->rows);
                scalarMultiply(dbi_avg[i], currentLearningRate / data->rows);
            }

            // add regularization
            for (i = 0; i < network->numConnections; i++){
                copyValuesInto(network->connections[i]->weights, regi[i]);
                scalarMultiply(regi[i], regularizationStrength);
                addTo(regi[i], dWi_avg[i]);
            }

            // add momentum
            for (i = 0; i < network->numConnections; i++){
                scalarMultiply(dWi_last[i], momentumFactor);
                scalarMultiply(dbi_last[i], momentumFactor);
                addTo(dWi_last[i], dWi_avg[i]);
                addTo(dbi_last[i], dbi_avg[i]);
            }

            // adjust weights and bias
            for (i = 0; i < network->numConnections; i++){
                scalarMultiply(dWi_avg[i], -1);
                scalarMultiply(dbi_avg[i], -1);
                addTo(dWi_avg[i], network->connections[i]->weights);
                addTo(dbi_avg[i], network->connections[i]->bias);
            }

            // cache weight and bias updates for momentum
            for (i = 0; i < network->numConnections; i++){
                copyValuesInto(dWi_avg[i], dWi_last[i]);
                copyValuesInto(dbi_avg[i], dbi_last[i]);
                // make positive again for next epoch
                scalarMultiply(dWi_last[i], -1);
                scalarMultiply(dbi_last[i], -1);
            }

            // zero out reusable average matrices and regularization matrices
            for (i = 0; i < network->numConnections; i++){
                zeroMatrix(dWi_avg[i]);
                zeroMatrix(dbi_avg[i]);
                zeroMatrix(regi[i]);
            }

            // free list of training examples
            for (i = 0; i < curBatchSize; i++){
                free(splitTraining[i]);
                free(splitClasses[i]);
            }
            free(splitTraining);
            free(splitClasses);

            // if verbose is set, print loss every 100 epochs
            if (verbose != 0){
                if (epoch % 100 == 0 || epoch == 1){
                    forwardPassDataSet(network, data);
                    if (lossFunction == CROSS_ENTROPY_LOSS){
                        printf("EPOCH %d: loss is %f\n", epoch, crossEntropyLoss(network, getOuput(network), classes, regularizationStrength));
                    }
                    else{
                        printf("EPOCH %d: loss is %f\n", epoch, meanSquaredError(network, getOuput(network), classes, regularizationStrength));
                    }
                }
            }
        }

        // free batch data for next batch creation
        for (i = 0; i < numBatches; i++){
            free(dataBatches[i]);
            free(classBatches[i]);
        }
        free(dataBatches);
        free(classBatches);
    }

    // free all reusable matrices
    destroyMatrix(beforeOutputT);
    for (i = 0; i < network->numConnections; i++){
        destroyMatrix(errori[i]);
        destroyMatrix(dWi[i]);
        destroyMatrix(dbi[i]);
    }
    destroyMatrix(errori[i]);

    for (i = 0; i < network->numConnections; i++){
        destroyMatrix(dWi_avg[i]);
        destroyMatrix(dbi_avg[i]);
        destroyMatrix(dWi_last[i]);
        destroyMatrix(dbi_last[i]);
        destroyMatrix(regi[i]);
    }

    if (numHidden > 0){
        for (i = 0; i < numHidden; i++){
            destroyMatrix(WTi[i]);
            destroyMatrix(errorLastTi[i]);
            destroyMatrix(fprimei[i]);
            destroyMatrix(inputTi[i]);
        }
        free(WTi);
        free(errorLastTi);
        free(fprimei);
        free(inputTi);
    }
}

#endif