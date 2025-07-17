// This is the practice for MatMul with **Tilting & Corner Turning**, for Chapter 06 HW1.
//
// Why corner turning? The matrix N is **stored** in **row-major**, but **accessed** in a **column-wise** way.
// This causes discrepency of data storage and access, and cannot benefit from memory coalescing.
//
// It can run in LeetGPU playground.

#include <iostream>
#include <cuda_runtime.h>

#include <assert.h>

// The width of the input 2d square matrix.
#define WIDTH 128
// Whether to turn on debugging print.
#define DEBUG false
// The size of block in 1 dim. Here the block is square shape.
#define BLOCK_SIZE 32

// Convert 2D index to flatten index for row major matrix.
__host__ __device__ int ToRowMajorFlattenIndex(int i, int j, int row_len) {
    return i * row_len + j;
}

// Convert 2D index to flatten index for column major matrix.
__host__ __device__ int ToColMajorFlattenIndex(int i, int j, int col_len) {
    return j * col_len + i;
}

// Convert 2D matrix from row major layout to column major layout.
void RowMajorToColumnMajor(float* matrix, int width) {
    float *matrix_copy = (float *)malloc(width * width * sizeof(float));
    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int flatten_idx = ToRowMajorFlattenIndex(row, col, width);
            matrix_copy[flatten_idx] = matrix[flatten_idx];
        }
    }

    for (int row = 0; row < width; row++) {
        for (int col = 0; col < width; col++) {
            int row_major_flatten_idx = ToRowMajorFlattenIndex(row, col, width);
            int col_major_flatten_idx = ToColMajorFlattenIndex(row, col, width);
            matrix[col_major_flatten_idx] = matrix_copy[row_major_flatten_idx];
        }
    }

    delete matrix_copy;
}

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
            Mds[ty][tx] = M[ToRowMajorFlattenIndex(m_row, m_col, Width)];
        } else {
            Mds[ty][tx] = 0.0;
        }
        
        if ((n_row < Width) && (n_col < Width)) {
            Nds[ty][tx] = N[ToColMajorFlattenIndex(n_row, n_col, Width)];
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

    // Make the N matrix to be col major layout for the purpose of corner turning.
    float b_h_col_major[WIDTH][WIDTH];
    Init2DSquareMatrix((float *)b_h_col_major, WIDTH);
    RowMajorToColumnMajor((float *)b_h_col_major, WIDTH);

    // Have another copy of N matrix but with row major layout, for the purpose of result verification.
    float b_h_row_major[WIDTH][WIDTH];
    Init2DSquareMatrix((float *)b_h_row_major, WIDTH);

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
    cudaMemcpy(b_d, b_h_col_major, size, cudaMemcpyHostToDevice);

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
    CpuMatMul((float*)a_h, (float*)b_h_row_major, (float*)expected, WIDTH);

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
