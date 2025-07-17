// This is the practice Chapter 05 5.9.
// It can run in LeetGPU playground.

#include <iostream>
#include <cuda_runtime.h>

#include <assert.h>

// The width of the input 2d square matrix.
#define WIDTH 256
// Whether to turn on debugging print.
#define DEBUG false
// The size of block in 1 dim. Here the block is square shape.
#define BLOCK_SIZE 32

// Each thread prodces one output matrix row.
__global__ void MatrixMulKernel(float* M, float* N, float* P, int Width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
    
    float PValue = 0.0;

    // "p" means phase.
    for (int ph = 0; ph <= int((Width - 1) / BLOCK_SIZE); ph++) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Load shared memory
        int m_row = row;
        int m_col = ph * BLOCK_SIZE + tx;
        int n_row = ph * BLOCK_SIZE + ty;
        int n_col = col;

        if ((m_row < Width) && (m_col < Width)) {
            Mds[ty][tx] = M[m_row * Width + m_col];
        } else {
            Mds[ty][tx] = 0.0;
        }
        
        if ((n_row < Width) && (n_col < Width)) {
            Nds[ty][tx] = N[n_row * Width + n_col];
        } else {
            Nds[ty][tx] = 0.0;
        }
        
        __syncthreads();

        // Computation.

        if ((row < Width) && (col < Width)) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                PValue += Mds[ty][k] * Nds[k][tx];
            }
        }
        __syncthreads();
    }
    

    // Computation
    if ((row < Width) && (col < Width)) {
        P[row * Width + col] = PValue;
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

// Initialize the values for 2D square matrix.
// E.g. when the width is 4, the output is
// {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}, {9.0, 10.0, 11.0, 12.0}, {13.0, 14.0, 15.0, 16.0}}
void Init2DSquareMatrix(float* a, int width) {
    int count = 0;
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            a[i * width + j] = ++count;
        }
    }
}

int main() {
    // The 2 input matrix.
    float a_h[WIDTH][WIDTH];
    Init2DSquareMatrix((float *)a_h, WIDTH);

    float b_h[WIDTH][WIDTH];
    Init2DSquareMatrix((float *)b_h, WIDTH);

    printf("Initialized the 2 input matrix...\n");

    // The result matrix.
    float c_h[WIDTH][WIDTH] = {0.0};

    // debug
    if (DEBUG) {
        for (int i = 0; i < WIDTH; i++) {
            for (int j = 0; j < WIDTH; j++) {
                printf("%f\n", a_h[i][j]);
            }
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
    int grid_dim_x = int((WIDTH - 1) / BLOCK_SIZE) + 1;
    printf("Starting kernel with grid dim [%d, %d, %d], block dim [%d, %d, %d]\n", grid_dim_x, grid_dim_x, 1, BLOCK_SIZE, BLOCK_SIZE, 1);
    MatrixMulKernel<<<dim3(grid_dim_x, grid_dim_x, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(a_d, b_d, c_d, WIDTH);

    // Copy from the device memory
    cudaMemcpy(c_h, c_d, size, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    printf("CUDA computation finished...\n");

    // Print result
    if (DEBUG) {
        printf("\nResult is:\n");
        for (int i = 0; i < WIDTH; i++) {
            for (int j = 0; j < WIDTH; j++) {
                printf("%f\n", c_h[i][j]);
            }
        }
    }

    // Assert result
    float expected[WIDTH][WIDTH];
    CpuMatMul((float*)a_h, (float*)b_h, (float*)expected, WIDTH);

    if (DEBUG) {
        printf("\nExpectation is:\n");
        for (int i = 0; i < WIDTH; i++) {
            for (int j = 0; j < WIDTH; j++) {
                printf("%f\n", expected[i][j]);
                assert(expected[i][j] == c_h[i][j]);
            }
        }
    }
    printf("CPU computation for verification finished...\n");

    printf("\nGood job! Your result is correct!!!\n\n");

    return 0;
}
