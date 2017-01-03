#include "std_includes.h"
#include "matrix.h"

#ifndef FUNCTION_H
#define FUNCTION_H

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

// raw tanh function
double tanHFunc(double input);

// derivatve of tanh given output of sigmoid
double tanHDeriv(double sigmoidInput);

// applies sigmoid function to each entry of $input
void sigmoid(Matrix* input);

// applies ReLU function to each entry of input
void relu(Matrix* input);

// applues tanh function to each entry of input
void tanH(Matrix* input);

// applies softmax function to each row of $input
void softmax(Matrix* input);

// sample from the unit guassian distribution (mean = 0, variance = 1)
double box_muller();


/*
    Begin functions.
*/

double sigmoidFunc(double input){
    return 1 / (1 + exp(-1 * input));
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

double tanHFunc(double input){
    return tanh(input);
}

double tanHDeriv(double tanhInput){
    return 1 - (tanhInput * tanhInput);
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
void tanH(Matrix* input){
    int i, j;
    for (i = 0; i < input->rows; i++){
        for (j = 0; j < input->cols; j++){
            input->data[i][j] = tanHFunc(input->data[i][j]); 
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

// adapted from wikipedia
double box_muller(){
    const double epsilon = DBL_MIN;
    const double two_pi = 2.0 * 3.14159265358979323846;
    static double z0, z1;
    static int generate;
    generate = generate == 1 ? 0 : 1;
    if (!generate){
        return z1;
    }
    double u1, u2;
    do{
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0;
}

#endif