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

__global__ void SetHeadIndex(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dheadbool);

__global__ void ReducePathLengths(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dlength,
                            int* dreducedlength);

__global__ void SetEdges(const int nrVertices,
                        int * dedgestatus,
                        int * ddegrees);

__global__ void AtomicallyNumberEachCompletePath(int nrVertices, 
                                                int *dbackwardlinkedlist, 
                                                int *dlength, 
                                                int *dheadindex,
                                                int *dfullpathcount,
                                                int* dsearchtree);

__global__ void CalculateNumberOfLeafNodes(
                                        int * dfullpathcount,
                                        int * dnumleaves);

__global__ void CalculateLeafOffsets(
                                        int * dfullpathcount,
                                        int * dnumleaves,
                                        int * active_leaf_offsets);
                                        

namespace mtc
{
class VCGPU
{
	public:
		VCGPU(const Graph &_graph, const int &_threadsPerBlock, const unsigned int &_barrier);
		~VCGPU();
        
        void numberCompletedPaths(int nrVertices, 
                                int *dbackwardlinkedlist, 
                                int *dlength);		
		
        void findCover(int nrVertices, 
                        int threadsPerBlock, 
                        int *dforwardlinkedlist, 
                        int *dbackwardlinkedlist, 
                        int *dmatch, 
                        int *dlength, 
                        int *dheadindex);		
                        
        void SortByHeadBool(int nrVertices,
                                int * dheadbool,
                                int * dheadboolSorted,
                                int * dheadlist,
                                int * dheadlistSorted);

        void GetLengthStatistics(int nrVertices, 
                                int threadsPerBlock, 
                                int *dbackwardlinkedlist, 
                                int *dlength,
                                int *dreducedlength);

        void GetDeviceVectors(int nrVertices, std::vector<int> & fll, std::vector<int> & bll, std::vector<int> & length);

        // VC arrays
        int *dedgestatus, *ddegrees, *dheadindex, *dfullpathcount, *active_leaf_offsets, *dsearchtree;

        int *dlength, *dforwardlinkedlist, *dbackwardlinkedlist;
        thrust::device_vector<int> dfll;
        thrust::device_vector<int> dbll;

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
