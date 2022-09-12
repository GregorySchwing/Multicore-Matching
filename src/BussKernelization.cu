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
__global__ void BussKernelizationP1Kernel(int nrVertices, 
                                        int k, 
                                        int *ddegrees,
                                        int *dsolution,
                                        int *dsizeofkernelsolution){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= nrVertices) return;
    int degree = ddegrees[threadID];
    if (degree <= k) return;
    printf("Vertex %d's deg %d exceeds k %d\n", threadID, degree, k);
    int solutionIndex = atomicAdd(&dsizeofkernelsolution[0], 1);
    // dsolution = new int[k];
    // Prevent oob
    if (solutionIndex >= k){
        return;
    }
    dsolution[solutionIndex] = threadID;
}

__global__ void BussKernelizationP2Kernel(int sizeOfKernelSolution,
                                        int *ddegrees,
                                        int *dremainingedges,
                                        int *dsolution){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= sizeOfKernelSolution) return;
    int solnVertex = dsolution[threadID];
    int degree = ddegrees[solnVertex];
    int remainingedges = atomicSub(&dremainingedges[0], degree);
    //printf("Removed %d's %d edges : edges remaining %d\n", solnVertex, degree/2, remainingedges/2);
}

BussKernelization::BussKernelization(const mtc::Graph &_graph, 
                                    const int &_threadsPerBlock, 
                                    const unsigned int &_barrier, 
                                    const unsigned int &_k,
                                    int *_dsolution,
                                    bool &_solutionCantExist):
    graph(_graph),
    threadsPerBlock(_threadsPerBlock),
    barrier(_barrier),
    k(_k),
    solutionCantExist(_solutionCantExist),
    dsolution(_dsolution){


    if (cudaMalloc(&ddegrees, sizeof(int)*graph.nrVertices) != cudaSuccess || 
        cudaMalloc(&dremainingedges, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dsizeofkernelsolution, sizeof(int)*1))	
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

// Initial kernelization before search tree is built
void BussKernelization::bussKernelizationP1(){
    cudaMemcpy(ddegrees, graph.degrees.data(), sizeof(int)*graph.nrVertices, cudaMemcpyHostToDevice);
    int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    BussKernelizationP1Kernel<<<blocksPerGrid, threadsPerBlock>>>(graph.nrVertices, 
                                                                k, 
                                                                ddegrees,
                                                                dsolution,
                                                                dsizeofkernelsolution);
    cudaMemcpy(&sizeOfKernelSolution, dsizeofkernelsolution, sizeof(int)*1, cudaMemcpyDeviceToHost);
    
    //cudaMemcpy(&solution[0], dsolution, sizeof(int)*sizeOfKernelSolution, cudaMemcpyDeviceToHost);
}

// Initial kernelization before search tree is built
void BussKernelization::bussKernelizationP2(){
    printf("Remaining edges before Kernel %d\n", graph.nrEdges);
    // Using the indices to calculate degrees requires doubling and then halving
    // Since each edge is counted twice, once in each connecting vertex's indices.x to indices.y
    remainingedges = 2*graph.nrEdges;
    cudaMemcpy(dremainingedges, &remainingedges, sizeof(int)*1, cudaMemcpyHostToDevice);
    int blocksPerGrid = (sizeOfKernelSolution + threadsPerBlock - 1)/threadsPerBlock;
    //printf("Launching %d blocks for a solution of size %d\n", blocksPerGrid, sizeOfKernelSolution);
    BussKernelizationP2Kernel<<<blocksPerGrid, threadsPerBlock>>>(sizeOfKernelSolution,
                                                                ddegrees,
                                                                dremainingedges,
                                                                dsolution);
    cudaMemcpy(&remainingedges, dremainingedges, sizeof(int)*1, cudaMemcpyDeviceToHost);
    // Using the indices to calculate degrees requires doubling and then halving
    // Since each edge is counted twice, once in each connecting vertex's indices.x to indices.y
    remainingedges/=2;
    printf("Remaining edges after Kernel %d\n", remainingedges);
}

// Initial kernelization before search tree is built
int BussKernelization::GetKPrime(){
    return kPrime;
}