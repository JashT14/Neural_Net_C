#ifndef NN_H
#define NN_H

#include <stdlib.h>

typedef struct {
    int num_layers;
    int *sizes; //neurons per layer
    double **biases; //biases[l][j]
    double **weights; //weights[l][j*prev+i]
} NN;

NN *nn_create(int num_layers,int *sizes);
void nn_free(NN *net);
void nn_forward(NN *net,double *input,double *output);
void nn_train(NN *net,double **inputs,double **targets,int n_samples,int epochs, double lr);
#endif
