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
#ifndef MATCH_MATCH_GPU_H
#define MATCH_MATCH_GPU_H

#include <vector>
#include <cuda.h>

#include "graph.h"
#include "matchcpu.h"

// For generalized MM heads & tails
#include<thrust/device_vector.h>
#include<thrust/sequence.h>
#include <thrust/fill.h>
namespace mtc
{

class GraphMatchingGPU : public GraphMatching
{
	public:
		GraphMatchingGPU(const Graph &, const int &, const unsigned int &);
		virtual ~GraphMatchingGPU();
		virtual void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution, int * kernelSolution, int leafIndex) const = 0;
		int2 *dneighbourRanges;
		int *dneighbours;
	protected:
		const int threadsPerBlock;
		const uint selectBarrier;
};

class GraphMatchingGPURandom : public GraphMatchingGPU
{
	public:
		GraphMatchingGPURandom(const Graph &, const int &, const unsigned int &);
		~GraphMatchingGPURandom();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;

};

class GraphMatchingGeneralGPURandom : public GraphMatchingGPU
{
	public:
		GraphMatchingGeneralGPURandom(const Graph &, const int &, const unsigned int &);
		~GraphMatchingGeneralGPURandom();

		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
		int *drequests, *dsense;
};

class GraphMatchingGPURandomMaximal : public GraphMatchingGPU
{
	public:
		GraphMatchingGPURandomMaximal(const Graph &, const int &, const unsigned int &);
		~GraphMatchingGPURandomMaximal();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

class GraphMatchingGPUWeighted : public GraphMatchingGPU
{
	public:
		GraphMatchingGPUWeighted(const Graph &, const int &, const unsigned int &);
		~GraphMatchingGPUWeighted();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;

	private:
		int *dweights;
};

class GraphMatchingGPUWeightedMaximal : public GraphMatchingGPU
{
	public:
		GraphMatchingGPUWeightedMaximal(const Graph &, const int &, const unsigned int &);
		~GraphMatchingGPUWeightedMaximal();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;

	private:
		int *dweights;
};

};

#endif
