#include "functions.h"

double sigmoidFunc(double input){
    return 1 / (1 + exp(input));
}

double sigmoidDeriv(double input){
    return sigmoidFunc(input) * (1 - sigmoidFunc(input));
}

// row vectors
void sigmoid(Matrix* input){
    assert(input->rows == 1);
    int i;
    for (i = 0; i < input->cols; i++){
        input->data[0][i] = sigmoidFunc(input->data[0][i]); 
    }
}

// row vectors
void softmax(Matrix* input){
    assert(input->rows == 1);
    double summed = 0;
    int i;
    for (i = 0; i < input->cols; i++){
        summed += exp(input->data[0][i]);
    }
    for (i = 0; i < input->cols; i++){
        input->data[0][i] = exp(input->data[0][i]) / summed; 
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
    return (-1 / actual->rows) * total_err;
}
