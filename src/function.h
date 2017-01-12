#include "std_includes.h"
#include "matrix.h"

#ifndef FUNCTION_H
#define FUNCTION_H

typedef void (*Activation)(Matrix*);

#define MAX(a,b) (((a)>(b))?(a):(b))

// raw sigmoid function
static float sigmoidFunc(float input);

// derivatve of sigmoid given output of sigmoid
static float sigmoidDeriv(float sigmoidInput);

// raw ReLU function
static float reluFunc(float input);

// derivative of ReLU given output of ReLU
static float reluDeriv(float reluInput);

// raw tanh function
static float tanHFunc(float input);

// derivatve of tanh given output of sigmoid
static float tanHDeriv(float sigmoidInput);

// applies sigmoid function to each entry of $input
static void sigmoid(Matrix* input);

// applies ReLU function to each entry of input
static void relu(Matrix* input);

// applues tanh function to each entry of input
static void tanH(Matrix* input);

// applies softmax function to each row of $input
static void softmax(Matrix* input);

// applies linear function to each row of $input
static void linear(Matrix* input);

// derivative of linear function given output of linear
static float linearDeriv(float linearInput);

// sample from the unit guassian distribution (mean = 0, variance = 1)
static float box_muller();

// return the string representation of activation function
static const char* getFunctionName(Activation func);

// return the activation function corresponding to a name
static Activation getFunctionByName(const char* name);

// return derivative of activation function
static float (*activationDerivative(Activation func))(float);


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
            setMatrix(input, i, j, sigmoidFunc(getMatrix(input, i, j)));
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
            setMatrix(input, i, j, reluFunc(getMatrix(input, i, j)));
        }
    }
}

// operates on each row
void tanH(Matrix* input){
    int i, j;
    for (i = 0; i < input->rows; i++){
        for (j = 0; j < input->cols; j++){
            setMatrix(input, i, j, tanHFunc(getMatrix(input, i, j)));
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
            summed += expf(getMatrix(input, i, j));
        }
        for (j = 0; j < input->cols; j++){
            setMatrix(input, i, j, expf(getMatrix(input, i, j)) / summed);
        }
    }
}

// operates on each row
void linear(Matrix* input){}

float linearDeriv(float linearInput){
    return 1;
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

const char* getFunctionName(Activation func){
    if (func == sigmoid){
        return "sigmoid";
    }
    else if (func == relu){
        return "relu";
    }
    else if (func == tanH){
        return "tanH";
    }
    else if (func == softmax){
        return "softmax";
    }
    else{
        return "linear";
    }
}

Activation getFunctionByName(const char* name){
    if (strcmp(name, "sigmoid") == 0){
        return sigmoid;
    }
    else if (strcmp(name, "relu") == 0){
        return relu;
    }
    else if (strcmp(name, "tanH") == 0){
        return tanH;
    }
    else if (strcmp(name, "softmax") == 0){
        return softmax;
    }
    else{
        return linear;
    }
}

float (*activationDerivative(Activation func))(float){
    if (func == sigmoid){
        return sigmoidDeriv;
    }
    else if (func == relu){
        return reluDeriv;
    }
    else if (func == tanH){
        return tanHDeriv;
    }
    else{
        return linearDeriv;
    }
}

#endif