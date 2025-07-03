// This is the practice Chapter 03 HW 1a.
// It can run in LeetGPU playground.

#include <iostream>
#include <cuda_runtime.h>

#include <assert.h>


// Each thread prodces one output matrix row.
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    for (int col = 0; col < Width; col++) {
        if ((row < Width) && (col < Width)) {
            float PValue = 0.0;
            for (int k = 0; k < Width; k++) {
                PValue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = PValue;
        }
    }
}

void CpuMatMul(float* m1, float* m2, float* result, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            result[i*width + j] = 0;

            for (int k = 0; k < width; k++) {
                result[i*width+j] += m1[i*width+k] * m2[k*width+j];
            }
        }
    }
}

int WIDTH = 4;

int main() {
    float a_h[4][4] = {{1.0, 2.0, 3.0, 4.0}, {5.0,6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}};
    float b_h[4][4] = {{1.0, 2.0, 3.0, 4.0}, {5.0,6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}};
    float c_h[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};

    // debug
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%f\n", a_h[i][j]);
        }
    }

    // Allocate device memory
    int size = WIDTH * WIDTH * sizeof(float);
    float *a_d, *b_d, *c_d;

    cudaMalloc((void **)&a_d, size);
    cudaMalloc((void **)&b_d, size);
    cudaMalloc((void **)&c_d, size);

    cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size, cudaMemcpyHostToDevice);

    // Call kernel
    // gridDim = 2, blockDim = 2
    MatrixMulKernel<<<2, 2>>>(a_d, b_d, c_d, WIDTH);

    // Copy from the device memory
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    // Print result
    printf("\nResult is:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%f\n", c_h[i][j]);
        }
    }

    // Assert result
    float expected[4][4] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    CpuMatMul((float*)a_h, (float*)b_h, (float*)expected, WIDTH);

    printf("\nExpectation is:\n");
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < WIDTH; j++) {
            printf("%f\n", expected[i][j]);
            assert(expected[i][j] == c_h[i][j]);
        }
    }

    return 0;
}
