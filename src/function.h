#include "std_includes.h"
#include "matrix.h"

#ifndef FUNCTION_H
#define FUNCTION_H

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

// raw sigmoid function
float sigmoidFunc(float input);

// derivatve of sigmoid given output of sigmoid
float sigmoidDeriv(float sigmoidInput);

// raw ReLU function
float reluFunc(float input);

// derivative of ReLU given output of ReLU
float reluDeriv(float reluInput);

// raw tanh function
float tanHFunc(float input);

// derivatve of tanh given output of sigmoid
float tanHDeriv(float sigmoidInput);

// applies sigmoid function to each entry of $input
void sigmoid(Matrix* input);

// applies ReLU function to each entry of input
void relu(Matrix* input);

// applues tanh function to each entry of input
void tanH(Matrix* input);

// applies softmax function to each row of $input
void softmax(Matrix* input);

// sample from the unit guassian distribution (mean = 0, variance = 1)
float box_muller();


/*
    Begin functions.
*/

float sigmoidFunc(float input){
    return 1 / (1 + expf(-1 * input));
}

float sigmoidDeriv(float sigmoidInput){
    return sigmoidInput * (1 - sigmoidInput);
}

float reluFunc(float input){
    return MAX(0, input);
}

float reluDeriv(float reluInput){
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

float tanHFunc(float input){
    return tanh(input);
}

float tanHDeriv(float tanhInput){
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
        float summed = 0;
        int j;
        for (j = 0; j < input->cols; j++){
            summed += expf(input->data[i][j]);
        }
        for (j = 0; j < input->cols; j++){
            input->data[i][j] = expf(input->data[i][j]) / summed; 
        }
    }
}

// adapted from wikipedia
float box_muller(){
    const float epsilon = FLT_MIN;
    const float two_pi = 2.0 * 3.14159265358979323846;
    static float z0, z1;
    static int generate;
    generate = generate == 1 ? 0 : 1;
    if (!generate){
        return z1;
    }
    float u1, u2;
    do{
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    } while (u1 <= epsilon);
    z0 = sqrt(-2.0 * logf(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * logf(u1)) * sin(two_pi * u2);
    return z0;
}

#endif