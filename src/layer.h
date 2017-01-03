#include "std_includes.h"
#include "matrix.h"
#include "function.h"

#ifndef LAYER_H
#define LAYER_H

typedef enum LAYER_TYPE_ {
    INPUT,
    HIDDEN,
    OUTPUT
} LAYER_TYPE;

// input matrix will continue to store values
// even after activation occurs, so it is best
// to think of it simply as a store rather than
// strictly as input
typedef struct Layer_ {
    LAYER_TYPE type;
    int size;
    void (*activation)(Matrix*);
    Matrix* input; // (num_examples x size)
} Layer;

typedef struct Connection_ {
    Layer* from;
    Layer* to;
    Matrix* weights; // (from_size x to_size)
    Matrix* bias; // (1 x to_size)
} Connection;

// returns layer given metadata and configuration
Layer* createLayer(LAYER_TYPE type, int size, void (*activation)(Matrix*));

// creates a connection and creates weights and bias matrices
Connection* createConnection(Layer* from, Layer* to);

// initializes weights and biases within connection
void initializeConnection(Connection* connection);

// applies activation function to each input in layer
void activateLayer(Layer* layer);

// frees layer and its input
void destroyLayer(Layer* layer);

// frees connection, its weights, and biases, but not its layers
void destroyConnection(Connection* connection);


/*
    Begin functions.
*/
Layer* createLayer(LAYER_TYPE type, int size, void (*activation)(Matrix*)){
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->type = type;
    layer->size = size;
    layer->activation = activation;
    double** row = (double**)malloc(sizeof(double*));
    row[0] = (double*)malloc(sizeof(double) * size);
    layer->input = createMatrix(1, size, row);
    return layer;
}

Connection* createConnection(Layer* from, Layer* to){
    Connection* connection = (Connection*)malloc(sizeof(Connection));
    connection->from = from;
    connection->to = to;
    double** weights_data = (double**)malloc(sizeof(double*) * from->size);
    int i;
    for (i = 0; i < from->size; i++){
        weights_data[i] = (double*)malloc(sizeof(double) * to->size);
    }
    connection->weights = createMatrix(from->size, to->size, weights_data);
    double** bias_data = (double**)malloc(sizeof(double*));
    bias_data[0] = (double*)malloc(sizeof(double) * to->size);
    connection->bias = createMatrix(1, to->size, bias_data);
    return connection;
}

// weights are guassian random using fan-in method
// biases are 0
void initializeConnection(Connection* connection){
    int i, j;
    for (i = 0; i < connection->bias->cols; i++){
        connection->bias->data[0][i] = 0;
    }
    srand(time(NULL));
    for (i = 0; i < connection->weights->rows; i++){
        for (j = 0; j < connection->weights->cols; j++){
            int neuronsIn = connection->weights->rows;
            connection->weights->data[i][j] = box_muller() / sqrt(neuronsIn);
        }
    }
}

// assuming input of layer is filled with raw input,
// calls activation function on each of them, and
// modifies in-place
void activateLayer(Layer* layer){
    if (layer->activation != NULL){
        layer->activation(layer->input);
    }
}

void destroyLayer(Layer* layer){
    destroyMatrix(layer->input);
    free(layer);
}

void destroyConnection(Connection* connection){
    destroyMatrix(connection->weights);
    destroyMatrix(connection->bias);
    free(connection);
}

#endif