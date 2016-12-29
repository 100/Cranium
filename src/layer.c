#include "layer.h"

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

// weights are random between -2 and 2
// biases are 0
void initializeConnection(Connection* connection){
    int i, j;
    for (i = 0; i < connection->bias->cols; i++){
        connection->bias->data[0][i] = 0;
    }
    srand(time(NULL));
    for (i = 0; i < connection->weights->rows; i++){
        for (j = 0; j < connection->weights->cols; j++){
            int random = rand();
            double val = ((random % 2 == 0 ? -1 : 1) * random) / (.5 * RAND_MAX);
            connection->weights->data[i][j] = val;
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