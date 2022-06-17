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
#include <memory>
// RE?
#include <ncurses.h>


#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "../DotWriter/lib/DotWriter.h"
#include "../DotWriter/lib/Enums.h"

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
                                    int depthOfSearchTree,
                                    int leafIndex,
                                    float * dfinishedLeavesPerLevel,
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

__global__ void CalculateDegrees(
                        int nrVertices,
                        int * dedgestatus,
                        int * ddegrees);

                             


class VCGPU
{

    class DotWriter;
	public:
		VCGPU(const mtc::Graph &_graph, const int &_threadsPerBlock, const unsigned int &_barrier, const unsigned int &_k);
		~VCGPU();
        long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
void writeGraphViz(std::vector<int> & match, 
					const Graph & g,
					const string &fileName_arg,  
					std::vector<int> & fll,
					std::vector<int> & bll);
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
        int fullpathcount, depthOfSearchTree;
        std::vector<float> finishedLeavesPerLevel;
        std::vector<float> totalLeavesPerLevel;
        std::vector<int> edgestatus;
        std::vector<int> newdegrees;
        std::vector<int2> searchtree;

        
        // VC arrays
        int *dedgestatus, *ddegrees, *dfullpathcount, *dnumleaves;
        int2 *dsearchtree;
        int *active_frontier_status;
        float * dfinishedLeavesPerLevel;

        int *dlength, *dforwardlinkedlist, *dbackwardlinkedlist;
        thrust::device_vector<int> dfll;
        thrust::device_vector<int> dbll;

        thrust::host_vector<int> recursive_leaf_offsets;
        thrust::device_vector<int> d_recursive_leaf_offsets;

        thrust::host_vector<int> recursive_leaf_counters;
        thrust::device_vector<int> d_recursive_leaf_counters;

        mtc::GraphMatchingGeneralGPURandom matcher;
	protected:
		const mtc::Graph &graph;
        const int &threadsPerBlock;
        const unsigned int &barrier;
		//int2 *dneighbourRanges;
		//int *dneighbours;
        
};

#endif
