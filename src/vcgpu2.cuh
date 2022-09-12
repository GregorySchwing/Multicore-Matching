/*
Copyright 2011, Bas Fagginger Auer.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
#ifndef MATCH_VC_GPU2_H
#define MATCH_VC_GPU2_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "graph.h"
#include "matchgpu.h"
#include <exception>
#include <memory>
// RE?
#include <ncurses.h>
#include "GrowthPolicy.h"
#include "TreePolicy.h"

#include "GraphViz.cuh"


// Library code
template <class GrowthPolicy, class TreePolicy>
class VCGPU2 : public GrowthPolicy, TreePolicy
{

	public:
		VCGPU2(const mtc::Graph &_graph, 
              const int &_threadsPerBlock, 
              const unsigned int &_barrier, 
              const unsigned int &_k,
              bool &solutionCantExist);
    void DoSomething();
		~VCGPU2();
        template <typename T> void myFunction(T* param1, int param2);

        bool & solutionCantExist;
        long long sizeOfSearchTree;
        int k;
        int kPrime;
        int fullpathcount, depthOfSearchTree, remainingedges;
        std::vector<float> finishedLeavesPerLevel;
        std::vector<float> totalLeavesPerLevel;
        std::vector<int> newdegrees;
        std::vector<int> solution;
        int solutionSize;
        std::vector<int> dynamcverts;
        int sizeOfKernelSolution;
        int numofdynamcverts;
        int numoftreeverts;
        int uncoverededges;
        int * duncoverededges;
        // VC arrays
        int *dedgestatus, *ddegrees, *dfullpathcount, *dnumleaves, *dremainingedges;
        int2 *dsearchtree;
        // Indicates the dyn added verts in each recursion stack
        int *ddynamicallyaddedvertices_csr;
        int *ddynamicallyaddedvertices;
        int *dnumberofdynamicallyaddedvertices;

        int *dsolution;
        int *dsizeofkernelsolution;
        int numberofdynamicallyaddedvertices;
        int numberofdynamicallyaddedverticesLB, numberofdynamicallyaddedverticesUB;
        int *active_frontier_status;
        float * dfinishedLeavesPerLevel;
        mtc::Edge * dedges;

        int *dlength, *dforwardlinkedlist, *dbackwardlinkedlist, *dmatch;


        mtc::GraphMatchingGeneralGPURandom matcher;

        GraphViz Gviz;
	protected:
		const mtc::Graph &graph;
        const int &threadsPerBlock;
        const unsigned int &barrier;
		//int2 *dneighbourRanges;
		//int *dneighbours;
        
};



using namespace std;
using namespace mtc;

inline void checkLastErrorCUDA(const char *file, int line)
{
  cudaError_t code = cudaGetLastError();
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    exit(code);
  }
}


//==== Kernel variables ====
__device__ int dkeepMatching;



template <class GrowthPolicy, class TreePolicy>
VCGPU2<GrowthPolicy, TreePolicy>::VCGPU2(const Graph &_graph, 
             const int &_threadsPerBlock, 
             const unsigned int &_barrier, 
             const unsigned int &_k,
             bool & _solutionCantExist):
		graph(_graph),
        threadsPerBlock(_threadsPerBlock),
        barrier(_barrier),
		matcher(_graph, _threadsPerBlock, _barrier),
        k(_k),
        solutionCantExist(_solutionCantExist)
{
    TreePolicy().Create();
}

template <class GrowthPolicy, class TreePolicy>
void VCGPU2<GrowthPolicy, TreePolicy>::DoSomething()
{
  TreePolicy().Create();
}

template <class GrowthPolicy, class TreePolicy>
VCGPU2<GrowthPolicy, TreePolicy>::~VCGPU2(){

}

template <class GrowthPolicy, class TreePolicy>
template <typename T>
void VCGPU2<GrowthPolicy, TreePolicy>::myFunction(T* param1, int param2){
    kernel_wrapper(param1, param2);
}

#endif
