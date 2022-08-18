#include <math.h>
#include <stdlib.h>
#include "image.h"
#include "matrix.h"

// 일기 : backward_layer() 와 update_layer()에서 free_matrix()를 적절한 위치에서 해줘야 하는데, 언제 해줘야 하는지 잘 모르겠다.
// 적절한 위치에서 free_matrix()를 해주지 않으면, 학습코드인 tryml.py 를 돌릴때, 자꾸 segmentation fault 가 뜬다. 
// 자세히 알려면, 구조체 변수인 layer 의 멤버변수인 matrix 구조체 변수 l->dw 에 새로만든 matrix 변수의 인스턴스인 dw 를 할당했을때, 
// 어떻게 코드가 돌아가는 지 그 상황을 이해해야 할 것 같다.

// Run an activation function on each element in a matrix,
// modifies the matrix in place
// matrix m: Input to activation function
// ACTIVATION a: function to run
void activate_matrix(matrix m, ACTIVATION a)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0.0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LOGISTIC){
                // TODO
                m.data[i][j] = (exp(x))/(1 + exp(x));
            } else if (a == RELU){
                // TODO
                if(x >= 0.0)
                {
                    m.data[i][j] = x;
                }
                else
                {
                    m.data[i][j] = 0.0;
                }
            } else if (a == LRELU){
                // TODO
                if(x >= 0.0)
                {
                    m.data[i][j] = x;
                }
                else
                {
                    m.data[i][j] = 0.01 * x;
                }
            } else if (a == SOFTMAX){
                // TODO
                m.data[i][j] = exp(x);
            }
            sum += m.data[i][j];
        }
        if (a == SOFTMAX) {
            // TODO: have to normalize by sum if we are using SOFTMAX
           for(j = 0; j < m.cols; j++){
                m.data[i][j] = m.data[i][j] / sum; 
           }
        }
    }
}

// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient

// a == SOFTMAX 는 여기서 하는게 아닌가?
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        double sum = 0.0;
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            // TODO: multiply the correct element of d by the gradient
            if(a == LOGISTIC){
                d.data[i][j] = d.data[i][j] * x * (1 - x);
            }
            else if(a == RELU){
                double tmp;
                if(m.data[i][j] > 0.0)
                {
                    tmp = 1.0;
                }
                else
                {
                    tmp = 0.0;
                }
                d.data[i][j] = d.data[i][j] * tmp;
            }
            else if(a == LRELU){
                double tmp;
                if(m.data[i][j] > 0.0)
                {
                    tmp = 1.0;
                }
                else
                {
                    tmp = 0.01;
                }
                d.data[i][j] = d.data[i][j] * tmp;
            }
            else if(a == SOFTMAX){
                sum = sum + d.data[i][j] * m.data[i][j];
            }
        }
        if(a == SOFTMAX){
            for(int j = 0; j < m.cols; ++j){
                d.data[i][j] = m.data[i][j] * d.data[i][j] - sum * m.data[i][j];
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    matrix out = matrix_mult_matrix(l->in, l->w);
    activate_matrix(out, l->activation);
    /*
    if(l->activation == LOGISTIC){
        for(int i = 0; i < mat_mul_mat.rows; i++){
            for(int j = 0; j < mat_mul_mat.cols; j++){
                out.data[i][j] = exp(mat_mul_mat.data[i][j]) / (1 + exp(mat_mul_mat.data[i][j]));
            }
        }
    }
    else if(l->activation == RELU){
        for(int i = 0; i < mat_mul_mat.rows; i++){
            for(int j = 0; j < mat_mul_mat.cols; j++){
                if(mat_mul_mat.data[i][j] > 0){
                    out.data[i][j] = mat_mul_mat.data[i][j];    
                }
                else{
                    out.data[i][j] = 0.0;
                }
            }
        }
    }
    else if(l->activation == LRELU){
        for(int i = 0; i < mat_mul_mat.rows; i++){
            for(int j = 0; j < mat_mul_mat.cols; j++){
                if(mat_mul_mat.data[i][j] > 0){
                    out.data[i][j] = mat_mul_mat.data[i][j];
                }
                else{
                    out.data[i][j] = 0.01 * mat_mul_mat.data[i][j];
                }
            }
        }
    }
    else if(l->activation == SOFTMAX){
        for(int i = 0; i < mat_mul_mat.rows; i++){
            double sum = 0.0;
            for(int j = 0; j < mat_mul_mat.cols; j++){
                sum = sum + exp(mat_mul_mat.data[i][j]);
            }
            for(int j = 0; j < mat_mul_mat.cols; j++){
                out.data[i][j] = mat_mul_mat.data[i][j] / sum;
            }
        }
    }
    */
    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation

    return out;
}

// Backward propagate derivatives through a layer
// layer *l: pointer to the layer
// matrix delta: partial derivative of loss w.r.t. output of layer
// returns: matrix, partial derivative of loss w.r.t. input to layer
matrix backward_layer(layer *l, matrix delta)
{
    
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta); // dL/dy 인 delta 를 dL/d(xw) 로 업데이트함.


    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix xT = transpose_matrix(l->in);
    matrix dw = matrix_mult_matrix(xT, delta); // replace this
    free_matrix(xT);
    l->dw = dw;
    //matrix dw = matrix_mult_matrix(transpose_matrix(l->in), delta); // 이렇게 하면 seg fault 뜸

    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    matrix wT = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, wT); // replace this
    free_matrix(wT);
    //matrix dx = matrix_mult_matrix(delta, transpose_matrix(l->w)); // 이렇게 하면 seg fault 뜸
    



    //// SOLUTION ////
    /*
    // 1.4.1
    // delta is dL/dy
    // TODO: modify it in place to be dL/d(xw)
    gradient_matrix(l->out, l->activation, delta);

    // 1.4.2
    // TODO: then calculate dL/dw and save it in l->dw
    free_matrix(l->dw);
    matrix inT = transpose_matrix(l->in);
    matrix dw = matrix_mult_matrix(inT, delta);
    //matrix dw = matrix_mult_matrix(transpose_matrix(l->in), delta);
    free_matrix(inT);
    l->dw = dw;


    // 1.4.3
    // TODO: finally, calculate dL/dx and return it.
    //matrix dx = matrix_mult_matrix(delta, transpose_matrix(l->w));
    matrix wT = transpose_matrix(l->w);
    matrix dx = matrix_mult_matrix(delta, wT);
    free_matrix(wT);
    */
    ///////////////////

    return dx;
}

// Update the weights at layer l
// layer *l: pointer to the layer
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_layer(layer *l, double rate, double momentum, double decay)
{
    /*
    // First Trial : Failed
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    
    
    matrix v = make_matrix(l->w.rows, l->w.cols);
    for(int i = 0; i < l->w.rows; ++i){
        for(int j = 0; j < l->w.cols; ++j){
            v.data[i][j] = l->dw.data[i][j] - (decay * l->w.data[i][j]) + (momentum * l->v.data[i][j]);
        }
    }
    free_matrix(l->v);
    l->v = v;
    free_matrix(v);
    
    // Δw_t : v
    // dL/dw_t : l->dw
    // λ : decay
    // w_t : l->w
    // m : momentum
    // Δw_{t-1} : l->v


    // Update l->w
    // w_{t+1} = w_t + ηΔw_t
    for(int i = 0; i < l->w.rows; ++i){
        for(int j = 0; j < l->w.cols; ++j)
        {
            l->w.data[i][j] = l->w.data[i][j] + rate * l->v.data[i][j];
        }
    }
    // η : rate
    */



    // Second Trial : 
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix tmp = axpy_matrix(-decay, l->w, l->dw);
    matrix v = axpy_matrix(momentum, l->v, tmp);
    free_matrix(l->v);
    l->v = v;

    // Update l->w
    matrix w = axpy_matrix(rate, v, l->w);
    free_matrix(l->w);
    l->w = w;

    // Remember to free any intermediate results to avoid memory leaks
    free_matrix(tmp);

    

    //// SOLUTION ////
    /*
    // TODO:
    // Calculate Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    // save it to l->v
    matrix temp = axpy_matrix(-decay, l->w, l->dw);
    matrix dwt = axpy_matrix(momentum, l->v, temp);
    free_matrix(l->v);
    l->v = dwt;

    // Update l->w
    matrix w_t1 = axpy_matrix(rate, dwt, l->w);
    free_matrix(l->w);
    l->w = w_t1;

    // Remember to free any intermediate results to avoid memory leaks
    free_matrix(temp)
    */
    ///////////////////

    // Remember to free any intermediate results to avoid memory leaks
    
}

// Make a new layer for our model
// int input: number of inputs to the layer
// int output: number of outputs from the layer
// ACTIVATION activation: the activation function to use
layer make_layer(int input, int output, ACTIVATION activation)
{
    layer l;
    l.in  = make_matrix(1,1);
    l.out = make_matrix(1,1);
    l.w   = random_matrix(input, output, sqrt(2./input));
    l.v   = make_matrix(input, output);
    l.dw  = make_matrix(input, output);
    l.activation = activation;
    return l;
}

// Run a model on input X
// model m: model to run
// matrix X: input to model
// returns: result matrix
matrix forward_model(model m, matrix X)
{
    int i;
    for(i = 0; i < m.n; ++i){
        X = forward_layer(m.layers + i, X);
    }
    return X;
}

// Run a model backward given gradient dL
// model m: model to run
// matrix dL: partial derivative of loss w.r.t. model output dL/dy
void backward_model(model m, matrix dL)
{
    matrix d = copy_matrix(dL);
    int i;
    for(i = m.n-1; i >= 0; --i){
        matrix prev = backward_layer(m.layers + i, d);
        free_matrix(d);
        d = prev;
    }
    free_matrix(d);
}

// Update the model weights
// model m: model to update
// double rate: learning rate
// double momentum: amount of momentum to use
// double decay: value for weight decay
void update_model(model m, double rate, double momentum, double decay)
{
    int i;
    for(i = 0; i < m.n; ++i){
        update_layer(m.layers + i, rate, momentum, decay);
    }
}

// Find the index of the maximum element in an array
// double *a: array
// int n: size of a, |a|
// returns: index of maximum element
int max_index(double *a, int n)
{
    if(n <= 0) return -1;
    int i;
    int max_i = 0;
    double max = a[0];
    for (i = 1; i < n; ++i) {
        if (a[i] > max){
            max = a[i];
            max_i = i;
        }
    }
    return max_i;
}

// Calculate the accuracy of a model on some data d
// model m: model to run
// data d: data to run on
// returns: accuracy, number correct / total
double accuracy_model(model m, data d)
{
    matrix p = forward_model(m, d.X);
    int i;
    int correct = 0;
    for(i = 0; i < d.y.rows; ++i){
        if(max_index(d.y.data[i], d.y.cols) == max_index(p.data[i], p.cols)) ++correct;
    }
    return (double)correct / d.y.rows;
}

// Calculate the cross-entropy loss for a set of predictions
// matrix y: the correct values
// matrix p: the predictions
// returns: average cross-entropy loss over data points, 1/n Σ(-ylog(p))
double cross_entropy_loss(matrix y, matrix p)
{
    int i, j;
    double sum = 0;
    for(i = 0; i < y.rows; ++i){
        for(j = 0; j < y.cols; ++j){
            sum += -y.data[i][j]*log(p.data[i][j]);
        }
    }
    return sum/y.rows;
}


// Train a model on a dataset using SGD
// model m: model to train
// data d: dataset to train on
// int batch: batch size for SGD
// int iters: number of iterations of SGD to run (i.e. how many batches)
// double rate: learning rate
// double momentum: momentum
// double decay: weight decay
void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)
{
    int e;
    for(e = 0; e < iters; ++e){
        data b = random_batch(d, batch);
        matrix p = forward_model(m, b.X);
        fprintf(stderr, "%06d: Loss: %f\n", e, cross_entropy_loss(b.y, p));
        matrix dL = axpy_matrix(-1, p, b.y); // partial derivative of loss dL/dy
        backward_model(m, dL);
        update_model(m, rate/batch, momentum, decay);
        free_matrix(dL);
        free_data(b);
    }
}
