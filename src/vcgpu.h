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
#ifndef MATCH_VC_GPU_H
#define MATCH_VC_GPU_H

#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "graph.h"
#include "matchgpu.h"
#include <exception>

__global__ void InitDegrees(const int nrVertices,
                            int * ddegrees);


__global__ void ReducePathLengths(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dlength,
                            int* dreducedlength);

__global__ void SetEdges(const int nrVertices,
                        int * dedgestatus,
                        int * ddegrees,
                        int2 *dsearchtree);

__global__ void PopulateSearchTree(int nrVertices, 
                                                int leafIndex,
                                                int *dforwardlinkedlist, 
                                                int *dbackwardlinkedlist, 
                                                int *dlength, 
                                                int *dfullpathcount,
                                                int2* dsearchtree);

__global__ void CalculateNumberOfLeafNodes(
                                        int * dfullpathcount,
                                        int * dnumleaves);

int4 CalculateLeafOffsets(  int leafIndex,
                            int fullpathcount);

__host__ __device__ long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
                             

namespace mtc
{
class VCGPU
{
	public:
		VCGPU(const Graph &_graph, const int &_threadsPerBlock, const unsigned int &_barrier, const unsigned int &_k);
		~VCGPU();
        
        int4 numberCompletedPaths(int nrVertices, 
                                int leafIndex,
                                int *dbackwardlinkedlist, 
                                int *dlength);		
		                       
        void GetLengthStatistics(int nrVertices, 
                                int threadsPerBlock, 
                                int *dbackwardlinkedlist, 
                                int *dlength,
                                int *dreducedlength);

        void SetEdgesOfLeaf(int leafIndex);
        void Match();
        void FindCover(int root);
        void ReinitializeArrays();
        void PrintData ();
        void GetDeviceVectors(int nrVertices, std::vector<int> & fll, std::vector<int> & bll, std::vector<int> & length);
        long long sizeOfSearchTree;
        int k;
        int fullpathcount;
        int *edgestatus, *newdegrees;
        
        // VC arrays
        int *dedgestatus, *ddegrees, *dfullpathcount, *dnumleaves;
        int2 *dsearchtree;
        int *active_frontier_status;

        int *dlength, *dforwardlinkedlist, *dbackwardlinkedlist;
        thrust::device_vector<int> dfll;
        thrust::device_vector<int> dbll;

        thrust::host_vector<int> recursive_leaf_offsets;
        thrust::device_vector<int> d_recursive_leaf_offsets;

        thrust::host_vector<int> recursive_leaf_counters;
        thrust::device_vector<int> d_recursive_leaf_counters;

        GraphMatchingGeneralGPURandom matcher;

	protected:
		const Graph &graph;
        const int &threadsPerBlock;
        const unsigned int &barrier;
		//int2 *dneighbourRanges;
		//int *dneighbours;
        
};

};

#endif
