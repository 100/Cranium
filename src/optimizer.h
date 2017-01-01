#include "std_includes.h"
#include "matrix.h"
#include "functions.h"
#include "layer.h"
#include "network.h"

#ifndef OPTIMIZER_H
#define OPTIMIZER_H

void batchGradientDescent(Network* network, Matrix* data, Matrix* classes, double learningRate, int maxIters, int verbose);


/*
    Begin functions.
*/

void batchGradientDescent(Network* network, Matrix* data, Matrix* classes, double learningRate, int maxIters, int verbose){
    assert(network->layers[0]->size == data->cols);
    assert(data->rows == classes->rows);
    assert(network->layers[network->numLayers - 1]->size == classes->cols);

    // since it creates batches deterministically, the data batches and 
    // the class batches will still align properly
    Matrix** dataBatches = createBatches(data, data->rows);
    Matrix** classBatches = createBatches(classes, data->rows);

    int training, epoch, layer, i, j, k;

    // these will be reused per training instance
    Matrix* errori[network->numLayers];
    Matrix* dWi[network->numConnections];
    Matrix* dbi[network->numConnections];
    for (i = 0; i < network->numConnections; i++){
        errori[i] = createMatrixZeroes(1, network->layers[i]->size);
        dWi[i] = createMatrixZeroes(network->connections[i]->weights->rows, network->connections[i]->weights->cols);
        dbi[i] = createMatrixZeroes(1, network->connections[i]->bias->cols);
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
    for (i = 0; i < network->numConnections; i++){
        dWi_avg[i] = createMatrixZeroes(network->connections[i]->weights->rows, network->connections[i]->weights->cols);
        dbi_avg[i] = createMatrixZeroes(1, network->connections[i]->bias->cols);
    }

    for (epoch = 1; epoch <= maxIters; epoch++){
        for (training = 0; training < data->rows; training++){
            // current data point to train on
            Matrix* example = dataBatches[training];
            Matrix* target = classBatches[training];
            
            // pass error forward
            forwardPass(network, example);
            
            // calculate each iteration of backpropagation
            for (layer = network->numLayers - 1; layer > 0; layer--){
                Layer* to = network->layers[layer];
                Connection* con = network->connections[layer - 1];
                if (layer == network->numLayers - 1){
                    // calculate output layer's error
                    copyValuesInto(to->input, errori[layer]);
                    for (j = 0; j < errori[layer]->cols; j++){
                        errori[layer]->data[0][j] -= target->data[0][j];
                    }
                    // calculate dWi and dbi
                    copyValuesInto(con->weights, dWi[layer - 1]);
                    for (i = 0; i < dWi[layer - 1]->rows; i++){
                        for (j = 0; j < dWi[layer - 1]->cols; j++){
                            dWi[layer - 1]->data[i][j] = errori[layer]->data[0][j] * con->from->input->data[0][i];
                        }
                    }
                    copyValuesInto(errori[layer], dbi[layer - 1]);
                }
                else{
                    // calculate error term for hidden layer
                    int hiddenLayer = layer - 1;
                    transposeInto(network->connections[layer]->weights, WTi[hiddenLayer]);
                    multiplyInto(errori[layer + 1], WTi[hiddenLayer], errorLastTi[hiddenLayer]);
                    copyValuesInto(con->to->input, fprimei[hiddenLayer]);
                    for (j = 0; i < fprimei[hiddenLayer]->cols; i++){
                        // NOTE: must use appropriate derivative
                        fprimei[hiddenLayer]->data[0][j] = sigmoidDeriv(fprimei[hiddenLayer]->data[0][j]);
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

        // adjust weights and bias
        for (i = 0; i < network->numConnections; i++){
            scalarMultiply(dWi_avg[i], -1 * learningRate * (1.0 / data->rows));
            scalarMultiply(dbi_avg[i], -1 * learningRate * (1.0 / data->rows));
        }
        for (i = 0; i < network->numConnections; i++){
            addTo(dWi_avg[i], network->connections[i]->weights);
            addTo(dbi_avg[i], network->connections[i]->bias);
        }

        // zero out reusable average matrices
        for (i = 0; i < network->numConnections; i++){
            zeroMatrix(dWi_avg[i]);
            zeroMatrix(dbi_avg[i]);
        }

        // if verbose is set, print loss every 50 epochs
        if (verbose != 0){
            if (epoch % 50 == 0){
                forwardPass(network, data);
                printf("EPOCH %d: loss is %f\n", epoch, crossEntropyLoss(network->layers[network->numLayers - 1]->input, classes));
            }
        }
    }

    // free all reusable matrices
    for (i = 0; i < network->numConnections; i++){
        destroyMatrix(errori[i]);
        destroyMatrix(dWi[i]);
        destroyMatrix(dbi[i]);
    }
    destroyMatrix(errori[i]);

    for (i = 0; i < network->numConnections; i++){
        destroyMatrix(dWi_avg[i]);
        destroyMatrix(dbi_avg[i]);
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

    for (i = 0; i < data->rows; i++){
        free(dataBatches[i]);
        free(classBatches[i]);
    }
    free(dataBatches);
    free(classBatches);
}

#endif