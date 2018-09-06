
#include <iostream>
#include <stdio.h>

const int Ax = 320;
const int Ay = 320;
const int Bx = 640;
const int By = 320;

const float aVal = 1;
const float bVal = 2;

const int ITER = 10;

const int TILE_WIDTH = 32;
//const int BLOCK_ROWS = 8;



__global__
void matMul(const float *matA, const float *matB, float *matC)   {
    int x = blockIdx.x * TILE_WIDTH;
    int y = blockIdx.y * TILE_WIDTH;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int widthA = Ax;
    int widthB = Bx;

    float C = 0;

#pragma unroll
    for (int a = y * widthA, b = x;
            a < y * widthA + widthA;
            a += TILE_WIDTH, b += TILE_WIDTH * widthB)   {

        // Two sub-matrices of A and B are loaded into shared memory
        // There is insufficient storage to load the entire block matrices
        // at once for large matrices.
        __shared__ float aMat[TILE_WIDTH][TILE_WIDTH];
        __shared__ float bMat[TILE_WIDTH][TILE_WIDTH];

        // Each thread assigns one element of each sub-matrix to shared memory.
        aMat[ty][tx] = matA[ty * widthA + a + tx];
        bMat[ty][tx] = matB[b + ty * widthB + tx];

        __syncthreads();

#pragma unroll
        for (int i = 0; i < TILE_WIDTH; i ++)   {
            C += aMat[ty][i] * bMat[i][tx];
        }

//	__syncthreads();
    }

    matC[(y + ty) * widthB + x + tx] = C;
}



int main()  {
    float *A, *B, *C;

    cudaMallocManaged(&A, Ax * Ay * sizeof(float));
    cudaMallocManaged(&B, Bx * By * sizeof(float));
    cudaMallocManaged(&C, Bx * Ay * sizeof(float));

    for (int i = 0; i < Ax * Ay; i ++)  {
        A[i] = aVal;
    }

    for (int i = 0; i < Bx * By; i ++)  {
        B[i] = bVal;
    }

    dim3 threads(TILE_WIDTH, TILE_WIDTH);
    dim3 grid(Bx / TILE_WIDTH, Ay / TILE_WIDTH);

    matMul<<<grid, threads>>>(A, B, C);

    cudaDeviceSynchronize();


    cudaEvent_t start;
    cudaEvent_t stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    for (int i = 0; i < ITER; ++i)  {
        matMul<<<grid, threads>>>(A, B, C);
    }

    cudaEventRecord(stop);

    cudaDeviceSynchronize();


    float elapsedTime = 0;

    cudaEventElapsedTime(&elapsedTime, start, stop); 

    double msecPerIter = elapsedTime / ((double)ITER);

    double flops = 2.0 * (double)Ax * (double)Ay * (double)Bx;
    double gflops = (flops * 1.0e-9f) / (msecPerIter / 1000.0);

    printf("Performance = %.2f GFLOPS\nTime = %.2f msec per matrix\nOperations = %.0f operations per matrix\n%u threads/block\n",
            gflops,
	    msecPerIter,
            flops,
            threads.x * threads.y);



    float maxErr = 0.0f;
    for (int i = 0; i < Bx * Ay; i ++)  {
        maxErr = fmax(maxErr, fabs(Ax * aVal * bVal - C[i]));
    }

    std::cout << "Total Error: " << maxErr << std::endl;

    return 0;
}
