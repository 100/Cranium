#include "functions.h"

double sigmoidFunc(double input){
    return 1 / (1 + exp(input));
}

double sigmoidDeriv(double input){
    return sigmoidFunc(input) * (1 - sigmoidFunc(input));
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
    return (-1 / actual->rows) * total_err;
}
