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
#include "GrowthPolicy.cuh"
#include "TreePolicy.h"

#include "GraphViz.cuh"


// Library code
template <class GrowthPolicy>
class VCGPU2 : public GrowthPolicy
{

	public:
		VCGPU2(const mtc::Graph &_graph, 
              const int &_threadsPerBlock, 
              const unsigned int &_barrier, 
              const unsigned int &_k,
              bool &solutionCantExist);
    void FindCover(int root,
                      int recursiveStackDepth,
                      bool & foundSolution);
		~VCGPU2();
        template <typename T> void myFunction(T* param1, int param2);

        bool & solutionCantExist;
        int k;
        std::vector<int> solution;
        int solutionSize;
        int uncoverededges;
        int * duncoverededges;
        int *dsolution;


        // Replace this with the csr offset v+1 in dynverts in treepolicy
        //int numberofdynamicallyaddedvertices;
        // VC arrays
        // Replace this with the csr offset v+1 in tree
        //int fullpathcount;
        // Replace this with the csr offset v+1 in tree in treepolicy
        //int *dfullpathcount;
        mtc::Edge * dedges;


        GraphViz Gviz;
        // Cant get this working
        //BussKernelization & bk;
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



template <class GrowthPolicy>
VCGPU2<GrowthPolicy>::VCGPU2(const Graph &_graph, 
             const int &_threadsPerBlock, 
             const unsigned int &_barrier, 
             const unsigned int &_k,
             bool & _solutionCantExist):
  graph(_graph),
  threadsPerBlock(_threadsPerBlock),
  barrier(_barrier),
  //matcher(_graph, _threadsPerBlock, _barrier),
  k(_k),
  solutionCantExist(_solutionCantExist),
  GrowthPolicy(_graph, _threadsPerBlock, _barrier, _k)
{
  if (cudaMalloc(&dsolution, sizeof(int)*_k) != cudaSuccess)
  {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}
  //bk = new BussKernelization(_graph, _threadsPerBlock, _barrier, _k, dsolution, _solutionCantExist);
  //TreePolicy().Create(bk->GetKPrime());
}

template <class GrowthPolicy>
void VCGPU2<GrowthPolicy>::FindCover(int root,
                      int recursiveStackDepth,
                      bool & foundSolution)
{
  GrowthPolicy::FindCover(root, recursiveStackDepth, foundSolution);
}

template <class GrowthPolicy>
VCGPU2<GrowthPolicy>::~VCGPU2(){

}

template <class GrowthPolicy>
template <typename T>
void VCGPU2<GrowthPolicy>::myFunction(T* param1, int param2){
    kernel_wrapper(param1, param2);
}

#endif
