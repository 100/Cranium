// struct and functions for a layer and connection, which holds a matrix of weights and biases
// use matrix multiplcation to not need neuron structs

#include "functions.c"
#include <time.h>

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
    Matrix* input; // matrix (num_examples x size)
} Layer;

typedef struct Connection_ {
    Layer* from;
    Layer* to;
    Matrix* weights;
    Matrix* bias;
} Connection;

Layer* createLayer(LAYER_TYPE type, int size, void (*activation)(Matrix*));

Connection* createConnection(Layer* from, Layer* to);

void initializeConnection(Connection* connection);

void activateLayer(Layer* layer);

void destroyLayer(Layer* layer);

void destroyConnection(Connection* connection);