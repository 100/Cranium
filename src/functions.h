#include "std_includes.h"
#include "matrix.h"

#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// raw sigmoid function
double sigmoidFunc(double input);

// derivatve of sigmoid given output of sigmoid
double sigmoidDeriv(double sigmoidInput);

// raw ReLU function
double reluFunc(double input);

// derivative of ReLU given output of ReLU
double reluDeriv(double reluInput);

// applies sigmoid function to each entry of $input
void sigmoid(Matrix* input);

// applies ReLU function to each entry of input
void relu(Matrix* input);

// applies softmax function to each row of $input
void softmax(Matrix* input);

// calculate the cross entropy loss between two datasets
double crossEntropyLoss(Matrix* prediction, Matrix* actual);


/*
    Begin functions.
*/

double sigmoidFunc(double input){
    return 1 / (1 + exp(input));
}

double sigmoidDeriv(double sigmoidInput){
    return sigmoidInput * (1 - sigmoidInput);
}

double reluFunc(double input){
    return MAX(0, input);
}

double reluDeriv(double reluInput){
    return reluInput > 0 ? 1 : 0;
}

// operates on each row
void sigmoid(Matrix* input){
    int i, j;
    for (i = 0; i < input->rows; i++){
        for (j = 0; j < input->cols; j++){
            input->data[i][j] = sigmoidFunc(input->data[i][j]); 
        }
    }
}

// operates on each row
void relu(Matrix* input){
    int i, j;
    for (i = 0; i < input->rows; i++){
        for (j = 0; j < input->cols; j++){
            input->data[i][j] = reluFunc(input->data[i][j]); 
        }
    }
}

// operates on each row
void softmax(Matrix* input){
    int i;
    for (i = 0; i < input->rows; i++){
        double summed = 0;
        int j;
        for (j = 0; j < input->cols; j++){
            summed += exp(input->data[i][j]);
        }
        for (j = 0; j < input->cols; j++){
            input->data[i][j] = exp(input->data[i][j]) / summed; 
        }
    }
}

// matrixes of size [num examples] x [num classes]
double crossEntropyLoss(Matrix* prediction, Matrix* actual){
    assert(prediction->rows == actual->rows);
    assert(prediction->cols == actual->cols);
    double total_err = 0;
    int i, j;
    for (i = 0; i < prediction->rows; i++){
        double cur_err = 0;
        for (j = 0; j < prediction->cols; j++){
            cur_err += actual->data[i][j] * log(MAX(DBL_MIN, prediction->data[i][j]));
        }
        total_err += cur_err;
    }
    return (-1.0 / actual->rows) * total_err;
}

#endif