#include "BussKernelization.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

inline void checkLastErrorCUDA(const char *file, int line)
{
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}

// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void PrintDegrees(int nrVertices, 
                            int *ddegrees){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= nrVertices) return;
    int degree = ddegrees[threadID];
    printf("Vertex %d's deg %d\n", threadID, degree);
}

// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void PrintRowCols(int k, 
                            int *ddegrees){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= k) return;
    int degree = ddegrees[threadID];
    printf("Vertex %d's rc %d\n", threadID, degree);
}

// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void BussKernelizationP1Kernel(int nrVertices, 
                                        int k, 
                                        int kPrime,
                                        int recursiveStackIndex, 
                                        int *ddegrees,
                                        int *dKernelSolutionRows,
                                        int *dKernelSolutionCols){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= nrVertices) return;
    int degree = ddegrees[threadID];
    if (degree <= kPrime) return;
    printf("Vertex %d's deg %d exceeds k %d\n", threadID, degree, k);
    int solutionIndex = atomicAdd(&dKernelSolutionRows[recursiveStackIndex], 1);
    // dsolution = new int[k];
    // Prevent oob
    if (solutionIndex >= k){
        return;
    }
    dKernelSolutionCols[solutionIndex] = threadID;
}

__global__ void ReduceDegrees(          int nrVertices,
                                        int *ddegrees,
                                        int *dremainingedges){
    extern __shared__ double temp[];
    int threadID = threadIdx.x;
    int vertexID = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadID >= nrVertices) return;

    // Load degrees into shared memory
    temp[threadID] = ddegrees[vertexID];
    // Warp reduce block of degrees
    for (int d=blockDim.x>>1; d>=1; d>>=1) {
        __syncthreads();
        if (threadID<d) temp[threadID] += temp[threadID+d];
    }
    // Add block reduced value to global value.
    if (threadIdx.x == 0) atomicAdd(&dremainingedges[0], temp[threadID]);

}

__global__ void BussKernelizationP2Kernel(int sizeOfKernelSolution,
                                        int *ddegrees,
                                        int *dremainingedges,
                                        int *dkernelsolution){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= sizeOfKernelSolution) return;
    int solnVertex = dkernelsolution[threadID];
    int degree = ddegrees[solnVertex];
    int remainingedges = atomicSub(&dremainingedges[0], degree);
    //printf("Removed %d's %d edges : edges remaining %d\n", solnVertex, degree/2, remainingedges/2);
}

/*
BussKernelization::BussKernelization(const mtc::Graph &_graph, 
                                    const int &_threadsPerBlock, 
                                    const unsigned int &_barrier, 
                                    const unsigned int &_k,
                                    bool &_solutionCantExist):
    graph(_graph),
    threadsPerBlock(_threadsPerBlock),
    barrier(_barrier),
    k(_k),
    solutionCantExist(_solutionCantExist){


    if (cudaMalloc(&dkernelsolution, sizeof(int)*k) != cudaSuccess || 
        cudaMalloc(&ddegrees, sizeof(int)*graph.nrVertices) != cudaSuccess || 
        cudaMalloc(&dremainingedges, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dsizeofkernelsolution, sizeof(int)*1) != cudaSuccess)	
    {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}

    sizeOfKernelSolution = 0;
    bussKernelizationP1();
    if (sizeOfKernelSolution > k){
        printf("|S| = b (%d) > k (%d), no solution exists\n", sizeOfKernelSolution, k);
    } else {
        printf("|S| = b (%d) <= k (%d), a solution may exist\n", sizeOfKernelSolution, k);
    }
    solutionCantExist = sizeOfKernelSolution > k;
    kPrime = k - sizeOfKernelSolution;
    if(!solutionCantExist){
        printf("Setting k' = k %d - b %d = %d\n", k, sizeOfKernelSolution, kPrime);
        bussKernelizationP2();
        solutionCantExist = remainingedges > k*kPrime;
        if(remainingedges > k*kPrime){
            printf("|G'(E)| (%d) > k (%d) * k' (%d) = %d, no solution exists\n",remainingedges, k, kPrime, k*kPrime);
        } else {
            printf("|G'(E)| (%d) <= k (%d) * k' (%d) = %d, a solution may exist\n",remainingedges, k, kPrime, k*kPrime);
        }
    }
}
*/

void BussKernelization::PerformBussKernelization(int nrVertices,
                                                int threadsPerBlock,
                                                const int k,
                                                int & kPrime,
                                                int recursiveStackIndex,
                                                int * ddegrees,
                                                int * dkernelsolutionrows,
                                                int * dkernelsolutioncols,
                                                int * deviceRemainingEdges,
                                                bool & solutionCantExist){
    int sizeOfKernelSolution = 0;
    bussKernelizationP1(nrVertices,
                        threadsPerBlock, 
                        k,
                        kPrime,
                        sizeOfKernelSolution,
                        recursiveStackIndex,
                        ddegrees,
                        dkernelsolutionrows,
                        dkernelsolutioncols);
    if (sizeOfKernelSolution > k){
        printf("|S| = b (%d) > k (%d), no solution exists\n", sizeOfKernelSolution, k);
    } else {
        printf("|S| = b (%d) <= k (%d), a solution may exist\n", sizeOfKernelSolution, k);
    }
    solutionCantExist = sizeOfKernelSolution > k;
    kPrime -= sizeOfKernelSolution;
    if(!solutionCantExist){
        printf("Setting k' = k %d - b %d = %d\n", k, sizeOfKernelSolution, kPrime);
        int remainingedges = 0;
        bussKernelizationP2(nrVertices,
                            threadsPerBlock, 
                            recursiveStackIndex,
                            remainingedges,
                            sizeOfKernelSolution,
                            ddegrees,
                            deviceRemainingEdges,
                            dkernelsolutionrows,
                            dkernelsolutioncols);
        solutionCantExist = remainingedges > k*kPrime;
        if(remainingedges > k*kPrime){
            printf("|G'(E)| (%d) > k (%d) * k' (%d) = %d, no solution exists\n",remainingedges, k, kPrime, k*kPrime);
        } else {
            printf("|G'(E)| (%d) <= k (%d) * k' (%d) = %d, a solution may exist\n",remainingedges, k, kPrime, k*kPrime);
        }
    }
}

// Initial kernelization before search tree is built
void BussKernelization::bussKernelizationP1(int nrVertices,
                                            int threadsPerBlock,
                                            int k,
                                            int kPrime,
                                            int & sizeOfKernelSolution,
                                            int recursiveStackIndex,
                                            int * dDegrees,
                                            int * dKernelSolutionRows,
                                            int * dKernelSolutionCols){
    // Copy last number of kernel soln vertices 
    int temp[2];
    //cudaMemcpy(&dKernelSolutionRows[recursiveStackIndex], &dKernelSolutionRows[recursiveStackIndex-1], sizeof(int)*1, cudaMemcpyDeviceToDevice);
    int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    
    PrintDegrees<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, 
                                                    dDegrees);
    PrintRowCols<<<blocksPerGrid, threadsPerBlock>>>(
                                                    k, 
                                                    &dKernelSolutionRows[recursiveStackIndex-1]);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);	

    int test;
    cudaMemcpy(&test, &dKernelSolutionRows[(recursiveStackIndex-1)], sizeof(int)*1, cudaMemcpyDeviceToHost);
    PrintRowCols<<<blocksPerGrid, threadsPerBlock>>>(
                                                    k, 
                                                    dKernelSolutionCols);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);	
    
    BussKernelizationP1Kernel<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, 
                                                                k, 
                                                                kPrime,
                                                                recursiveStackIndex,
                                                                dDegrees,
                                                                dKernelSolutionRows,
                                                                dKernelSolutionCols);

    
    cudaMemcpy(temp, &dKernelSolutionRows[recursiveStackIndex-1], sizeof(int)*2, cudaMemcpyDeviceToHost);
    sizeOfKernelSolution = temp[1] - temp[0];
}

// Initial kernelization before search tree is built
void BussKernelization::bussKernelizationP2(int nrVertices,
                                        int threadsPerBlock,
                                        int recursiveStackIndex,
                                        int & remainingEdges,
                                        int sizeOfKernelSolution,
                                        int * dDegrees,
                                        int * deviceRemainingEdges,
                                        int * dkernelsolutionrows,
                                        int * dkernelsolutioncols)
{
    int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    ReduceDegrees<<<blocksPerGrid, threadsPerBlock>>>(nrVertices,
                                                      dDegrees,
                                                      deviceRemainingEdges);
    cudaMemcpy(&remainingEdges, &deviceRemainingEdges, sizeof(int)*1, cudaMemcpyDeviceToHost);
    printf("Remaining edges before Kernel %d\n", remainingEdges);
    // Using the indices to calculate degrees requires doubling and then halving
    // Since each edge is counted twice, once in each connecting vertex's indices.x to indices.y
    //remainingedges = 2*graph.nrEdges;
    //cudaMemcpy(dremainingedges, &remainingedges, sizeof(int)*1, cudaMemcpyHostToDevice);
    blocksPerGrid = (sizeOfKernelSolution + threadsPerBlock - 1)/threadsPerBlock;
    //printf("Launching %d blocks for a solution of size %d\n", blocksPerGrid, sizeOfKernelSolution);
    BussKernelizationP2Kernel<<<blocksPerGrid, threadsPerBlock>>>(sizeOfKernelSolution,
                                                                dDegrees,
                                                                deviceRemainingEdges,
                                                                &dkernelsolutioncols[dkernelsolutionrows[recursiveStackIndex]]);
    cudaMemcpy(&remainingEdges, deviceRemainingEdges, sizeof(int)*1, cudaMemcpyDeviceToHost);
    // Using the indices to calculate degrees requires doubling and then halving
    // Since each edge is counted twice, once in each connecting vertex's indices.x to indices.y
    remainingEdges/=2;
    printf("Remaining edges after Kernel %d\n", remainingEdges);
    
}

// Initial kernelization before search tree is built
/*
int BussKernelization::GetKPrime(){
    return kPrime;
}

// Initial kernelization before search tree is built
int* BussKernelization::GetKernelSolution(){
    return dkernelsolution;
}
*/