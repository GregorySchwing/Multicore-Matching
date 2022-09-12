#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "graph.h"

__global__ void BussKernelizationP1Kernel(int nrVertices, 
                                        int k, 
                                        int *ddegrees,
                                        int *dsolution,
                                        int *dsizeofkernelsolution);

__global__ void BussKernelizationP2Kernel(int sizeOfKernelSolution,
                                        int *ddegrees,
                                        int *dremainingedges,
                                        int *dsolution);

class BussKernelization
{

	public:
		BussKernelization(const mtc::Graph &_graph, 
                        const int &_threadsPerBlock, 
                        const unsigned int &_barrier, 
                        const unsigned int &_k,
                        int *_dsolution,
                        bool &solutionCantExist);
		~BussKernelization();
        void bussKernelizationP1();
        void bussKernelizationP2();
        int GetKPrime();
    private:
        int sizeOfKernelSolution;

        int *dremainingedges, *dsizeofkernelsolution, *ddegrees, *dsolution;
        bool solutionCantExist;
        int k;
        int kPrime;
        int remainingedges;
	protected:
		const mtc::Graph &graph;
        const int &threadsPerBlock;
        const unsigned int &barrier;
		//int2 *dneighbourRanges;
		//int *dneighbours;

};