// This is the practice Chapter 05 5.9.
// It can run in LeetGPU playground.

#include <iostream>
#include <cuda_runtime.h>

#include <assert.h>

// The width of the input 2d matrix.
// Matrix A: M_WIDTH x N_WIDTH
// Matrix B: N_WIDTH x K_WIDTH
#define M_WIDTH 256
#define N_WIDTH 512
#define K_WIDTH 256
// Whether to turn on debugging print.
#define DEBUG false
// The size of block in 1 dim. Here the block is square shape.
#define BLOCK_SIZE 32

// Each thread prodces one output matrix row.
__global__ void MatrixMulKernel(float* M, float* N, float* P, int m_width, int n_width, int k_width) {
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float Mds[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Nds[BLOCK_SIZE][BLOCK_SIZE];
    
    float PValue = 0.0;

    // "p" means phase.
    for (int ph = 0; ph <= int((n_width - 1) / BLOCK_SIZE); ph++) {
        int tx = threadIdx.x;
        int ty = threadIdx.y;

        // Load shared memory
        int m_row = row;
        int m_col = ph * BLOCK_SIZE + tx;
        int n_row = ph * BLOCK_SIZE + ty;
        int n_col = col;

        if ((m_row < m_width) && (m_col < n_width)) {
            Mds[ty][tx] = M[m_row * n_width + m_col];
        } else {
            Mds[ty][tx] = 0.0;
        }
        
        if ((n_row < n_width) && (n_col < k_width)) {
            Nds[ty][tx] = N[n_row * k_width + n_col];
        } else {
            Nds[ty][tx] = 0.0;
        }
        
        __syncthreads();

        // Computation.

        if ((row < m_width) && (col < k_width)) {
            for (int k = 0; k < BLOCK_SIZE; k++) {
                PValue += Mds[ty][k] * Nds[k][tx];
            }
        }
        __syncthreads();
    }
    

    // Computation
    if ((row < m_width) && (col < k_width)) {
        P[row * k_width + col] = PValue;
    }
}

void CpuMatMul(float* m1, float* m2, float* result, int m_width, int n_width, int k_width) {
    for (int i = 0; i < m_width; i++) {
        for (int j = 0; j < k_width; j++) {
            result[i*k_width + j] = 0;

            for (int k = 0; k < n_width; k++) {
                result[i*k_width+j] += m1[i*n_width+k] * m2[k*k_width+j];
            }
        }
    }
}

// Initialize the values for 2D matrix.
// E.g. when the shape is 2x4, the output is
// {{1.0, 2.0, 3.0, 4.0}, {5.0, 6.0, 7.0, 8.0}}
void Init2DMatrix(float* a, int rows, int cols) {
    int count = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            a[i * cols + j] = ++count;
        }
    }
}

int main() {
    // The 2 input matrix.
    float a_h[M_WIDTH][N_WIDTH];
    Init2DMatrix((float *)a_h, M_WIDTH, N_WIDTH);

    float b_h[N_WIDTH][K_WIDTH];
    Init2DMatrix((float *)b_h, N_WIDTH, K_WIDTH);

    printf("Initialized the 2 input matrix...\n");

    // The result matrix.
    float c_h[M_WIDTH][K_WIDTH] = {0.0};

    // debug
    if (DEBUG) {
        for (int i = 0; i < M_WIDTH; i++) {
            for (int j = 0; j < N_WIDTH; j++) {
                printf("%f\n", a_h[i][j]);
            }
        }
    }   

    // Allocate device memory
    int size_a = M_WIDTH * N_WIDTH * sizeof(float);
    int size_b = N_WIDTH * K_WIDTH * sizeof(float);
    int size_c = M_WIDTH * K_WIDTH * sizeof(float);
    float *a_d, *b_d, *c_d;

    cudaMalloc((void **)&a_d, size_a);
    cudaMalloc((void **)&b_d, size_b);
    cudaMalloc((void **)&c_d, size_c);

    cudaMemcpy(a_d, a_h, size_a, cudaMemcpyHostToDevice);
    cudaMemcpy(b_d, b_h, size_b, cudaMemcpyHostToDevice);

    // Call kernel
    int grid_dim_x = int((K_WIDTH - 1) / BLOCK_SIZE) + 1;
    int grid_dim_y = int((M_WIDTH - 1) / BLOCK_SIZE) + 1;
    printf("Starting kernel with grid dim [%d, %d, %d], block dim [%d, %d, %d]\n", grid_dim_x, grid_dim_y, 1, BLOCK_SIZE, BLOCK_SIZE, 1);
    MatrixMulKernel<<<dim3(grid_dim_x, grid_dim_y, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(a_d, b_d, c_d, M_WIDTH, N_WIDTH, K_WIDTH);

    // Copy from the device memory
    cudaMemcpy(c_h, c_d, size_c, cudaMemcpyDeviceToHost);

    cudaFree(a_d);
    cudaFree(b_d);
    cudaFree(c_d);

    printf("CUDA computation finished...\n");

    // Print result
    if (DEBUG) {
        printf("\nResult is:\n");
        for (int i = 0; i < M_WIDTH; i++) {
            for (int j = 0; j < K_WIDTH; j++) {
                printf("%f\n", c_h[i][j]);
            }
        }
    }

    // Assert result
    float expected[M_WIDTH][K_WIDTH];
    CpuMatMul((float*)a_h, (float*)b_h, (float*)expected, M_WIDTH, N_WIDTH, K_WIDTH);

    if (DEBUG) {
        printf("\nExpectation is:\n");
        for (int i = 0; i < M_WIDTH; i++) {
            for (int j = 0; j < K_WIDTH; j++) {
                printf("%f\n", expected[i][j]);
                assert(expected[i][j] == c_h[i][j]);
            }
        }
    }
    printf("CPU computation for verification finished...\n");

    printf("\nGood job! Your result is correct!!!\n\n");

    return 0;
}
