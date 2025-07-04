// This is the practice Chapter 03 HW 2.
// It can run in LeetGPU playground.

#include <iostream>
#include <cuda_runtime.h>

#include <assert.h>


// Each thread prodces one output vector element.
// M: a square matrix
// N: vector
// P: the output vector
// Width: the width of the square M, and the length of N & P
__global__ void MatVecMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if ((row < Width)) {
        float PValue = 0.0;
        for (int k = 0; k < Width; k++) {
            PValue += M[row * Width + k] * N[k];
        }
        P[row] = PValue;
    }
}

void CpuMatVecMul(float* m1, float* m2, float* result, int width) {
    for (int i = 0; i < width; i++) {
        result[i] = 0;

        for (int k = 0; k < width; k++) {
            result[i] += m1[i*width+k] * m2[k];
        }
    }
}

int WIDTH = 4;

int main() {
    float a_h[4][4] = {{1.0, 2.0, 3.0, 4.0}, {5.0,6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}};
    float b_h[4] = {1.0, 2.0, 3.0, 4.0};
    float c_h[4] = {0.0, 0.0, 0.0, 0.0};

    // debug
    for (int i = 0; i < WIDTH; i++) {
        printf("%f\n", c_h[i]);
    }

    // Allocate device memory
    int size_m = WIDTH * WIDTH * sizeof(float);
    int size_v = WIDTH * sizeof(float);
    float *a_d, *b_d, *c_d;

    cudaMalloc((void **)&a_d, size_m);
    cudaMalloc((void **)&b_d, size_v);
    cudaMalloc((void **)&c_d, size_v);

    cudaMemcpy(a_d, a_h, size_m, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size_v, cudaMemcpyHostToDevice);

    // Call kernel
    // gridDim = 2, blockDim = 2
    MatVecMulKernel<<<2, 2>>>(a_d, b_d, c_d, WIDTH);

    // Copy from the device memory
    cudaMemcpy(c_h, c_d, size_v, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // Print result
    printf("\nResult is:\n");
    for (int i = 0; i < WIDTH; i++) {
        printf("%f\n", c_h[i]);
    }

    // Assert result
    float expected[4] = {0.0, 0.0, 0.0, 0.0};
    CpuMatVecMul((float*)a_h, (float*)b_h, (float*)expected, WIDTH);

    printf("\nExpectation is:\n");
    for (int i = 0; i < WIDTH; i++) {
        printf("%f\n", expected[i]);
        assert(expected[i] == c_h[i]);
    }

    return 0;
}
