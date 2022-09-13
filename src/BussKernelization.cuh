/*
Copyright 2022, Gregory Schwing.

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
#ifndef BUSS_KERNELIZATION_H
#define BUSS_KERNELIZATION_H


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
    /*
		BussKernelization(const mtc::Graph &_graph, 
                        const int &_threadsPerBlock, 
                        const unsigned int &_barrier, 
                        const unsigned int &_k,
                        bool &solutionCantExist);
		~BussKernelization();
        */
        static void bussKernelizationP1(int nrVertices,
                                int threadsPerBlock,
                                const int k,
                                int kPrime,
                                int & sizeOfKernelSolution,
                                int recursiveStackIndex,
                                int * dDegrees,
                                int * dKernelSolutionRows,
                                int * dKernelSolutionCols);
        static void bussKernelizationP2(int nrVertices,
                                        int threadsPerBlock,
                                        int recursiveStackIndex,
                                        int & remainingEdges,
                                        int sizeOfKernelSolution,
                                        int * dDegrees,
                                        int * deviceRemainingEdges,
                                        int startOfNewKernel,
                                        int * dkernelsolutioncols);
        //int GetKPrime();
        //int* GetKernelSolution();
        static void PerformBussKernelization(int nrVertices,
                                                int threadsPerBlock,
                                                const int k,
                                                int & kPrime,
                                                int recursiveStackIndex,
                                                int * ddegrees,
                                                int * dkernelsolutionrows,
                                                int * dkernelsolutioncols,
                                                int * deviceRemainingEdges,
                                                bool & solutionCantExist);
                                                        /*
    private:
        int sizeOfKernelSolution;
        int *dkernelsolution;
        int *dremainingedges, *dsizeofkernelsolution, *ddegrees;
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
    */
};

#endif