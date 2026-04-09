#include "nn.h"
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <time.h>

static double sigmoid(double x){
    return 1.0/(1.0+exp(-x));
}
static double sigmoid_d(double s){ 
    return s*(1.0-s);
}

//random in [-1, 1]
static double rand_weight(void){
    return ((double)rand()/RAND_MAX)*2.0 - 1.0;
}

NN *nn_create(int num_layers,int *sizes){
    srand((unsigned)time(NULL));
    NN *net=(NN *)malloc(sizeof(NN));
    net->num_layers=num_layers;
    net->sizes=(int *)malloc(num_layers * sizeof(int));
    memcpy(net->sizes, sizes, num_layers * sizeof(int));

    //layers 1..n-1 have biases and weights
    net->biases=(double **)malloc(num_layers * sizeof(double *));
    net->weights=(double **)malloc(num_layers * sizeof(double *));
    net->biases[0]=NULL;
    net->weights[0]=NULL;

    for (int l=1;l<num_layers;l++) {
        int cur=sizes[l], prev=sizes[l-1];

        net->biases[l]=(double *)malloc(cur * sizeof(double));
        for (int j=0;j<cur;j++)
            net->biases[l][j]=rand_weight();

        net->weights[l]=(double *)malloc(cur * prev * sizeof(double));
        for (int j=0;j<cur*prev;j++)
            net->weights[l][j]=rand_weight();
    }
    return net;
}

void nn_free(NN *net){
    for (int l=1; l<net->num_layers;l++){
        free(net->biases[l]);
        free(net->weights[l]);
    }
    free(net->biases);
    free(net->weights);
    free(net->sizes);
    free(net);
}

// forward pass, stores activations per layer into `acts` if non-NULL
static void forward(NN *net, double *input, double **acts){
    int n=net->num_layers;
    memcpy(acts[0], input, net->sizes[0] * sizeof(double));
    for (int l=1;l<n;l++){
        int cur=net->sizes[l],prev=net->sizes[l - 1];
        for (int j=0; j<cur; j++){
            double sum=net->biases[l][j];
            for (int i=0;i<prev;i++)
                sum+=net->weights[l][j * prev + i] * acts[l - 1][i];
            acts[l][j]=sigmoid(sum);
        }
    }
}

void nn_forward(NN *net, double *input, double *output){
    int n=net->num_layers;

    // allocate temp activations
    double **acts=(double **)malloc(n * sizeof(double *));
    for (int l=0;l<n;l++)
        acts[l]=(double *)malloc(net->sizes[l] * sizeof(double));

    forward(net, input, acts);
    memcpy(output,acts[n-1],net->sizes[n-1] * sizeof(double));
    for (int l=0;l<n;l++) free(acts[l]);
    free(acts);
}

void nn_train(NN *net,double **inputs,double **targets,int n_samples,int epochs, double lr) {
    int n=net->num_layers;

    // allocate activations and deltas
    double **acts=(double **)malloc(n * sizeof(double *));
    double **deltas=(double **)malloc(n * sizeof(double *));
    for (int l=0;l<n;l++){
        acts[l]=(double *)malloc(net->sizes[l] * sizeof(double));
        deltas[l]=(double *)malloc(net->sizes[l] * sizeof(double));
    }

    for (int e=0; e<epochs; e++){
        double total_loss=0.0;

        for (int s=0; s<n_samples; s++){
            forward(net, inputs[s], acts);

            // output layer delta
            int last=n-1;
            for (int j=0;j<net->sizes[last];j++){
                double err=targets[s][j]-acts[last][j];
                deltas[last][j]=err * sigmoid_d(acts[last][j]);
                total_loss+=err * err;
            }

            // backprop hidden layers
            for (int l=last-1;l>=1;l--){
                int cur=net->sizes[l],next=net->sizes[l+1];
                for (int j=0;j<cur;j++){
                    double sum=0.0;
                    for (int k=0; k<next; k++)
                        sum+=deltas[l + 1][k] * net->weights[l + 1][k * cur + j];
                    deltas[l][j] = sum * sigmoid_d(acts[l][j]);
                }
            }

            // update weights and biases
            for (int l=1; l<n; l++) {
                int cur=net->sizes[l], prev = net->sizes[l - 1];
                for (int j=0; j<cur; j++){
                    net->biases[l][j]+=lr * deltas[l][j];
                    for (int i=0; i<prev; i++)
                        net->weights[l][j * prev + i] += lr * deltas[l][j] * acts[l - 1][i];
                }
            }
        }

        // print loss every 1000 epochs
        if ((e+1)%1000==0)
            printf("epoch %5d | loss: %.6f\n", e + 1, total_loss / n_samples);
    }

    for (int l = 0; l < n; l++) { free(acts[l]); free(deltas[l]); }
    free(acts);
    free(deltas);
}
