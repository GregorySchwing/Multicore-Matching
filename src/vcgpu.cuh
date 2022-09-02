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

#include "GraphViz.cuh"

__global__ void InitDegrees(const int nrVertices,
                            int * ddegrees);


__global__ void BussKernelizationP1Kernel(int nrVertices, 
                                        int k, 
                                        int *ddegrees,
                                        int *dsolution,
                                        int *dsizeofkernelsolution);

__global__ void BussKernelizationP2Kernel(int sizeOfKernelSolution,
                                        int *ddegrees,
                                        int *dremainingedges,
                                        int *dsolution);

__global__ void ReducePathLengths(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dlength,
                            int* dreducedlength);

__global__ void ReduceEdgeStatusArray(int nrNeighbors,
							int *dedgestatus,
                            int* dremainingedges);
                         
__global__ void SetEdges(const int nrVertices,
                        int * dedgestatus,
                        int * ddegrees,
                        int2 *dbfssearchtree);

__global__ void PopulateSearchTree(int nrVertices, 
                                    int sizeOfSearchTree,
                                    int depthOfSearchTree,
                                    int leafIndex,
                                    float * dfinishedLeavesPerLevel,
                                    int *dforwardlinkedlist, 
                                    int *dbackwardlinkedlist, 
                                    int *dlength, 
                                    int *dfullpathcount,
                                    int2* dbfssearchtree);

__global__ void PopulateSearchTreeTest(int nrVertices, 
                                    int sizeOfSearchTree,
                                    int depthOfSearchTree,
                                    int leafIndex,
                                    float * dfinishedLeavesPerLevel,
                                    int *dforwardlinkedlist, 
                                    int *dbackwardlinkedlist, 
                                    int *dlength, 
                                    int *dfullpathcount,
                                    int2* dbfssearchtree,
                                    int fullpathcount);

__global__ void CalculateNumberOfLeafNodes(
                                        int * dfullpathcount,
                                        int * dnumleaves);

__global__ void EvaluateSingleLeafNode(int nrEdges,
                                    int leafIndex,
                                    int depthOfLeaf,
                                    int sizeOfKernelSolution,
                                    int * dsolution,
                                    mtc::Edge * dedges, 
                                    int2 * dbfssearchtree,
                                    int * dnumberofdynamicallyaddedvertices,
                                    int * ddynamicallyaddedvertices,
                                    int * uncoverededges);

__global__ void eraseDynVertsOfRecursionLevel(int recursiveStackDepth,
                                              int * dnumberofdynamicallyaddedvertices, 
                                              int * ddynamicallyaddedvertices_csr, 
                                              int * ddynamicallyaddedvertices);

__global__ void CalculateDegrees(
                        int nrVertices,
                        int * dedgestatus,
                        int * ddegrees);

__global__ void DetectAndSetPendantPathsCase3(int nrVertices, 
                                                int k,
                                                int *match, 
                                                int *dforwardlinkedlist, 
                                                int *dbackwardlinkedlist, 
                                                int * dedgestatus,
                                                int *dlength, 
                                                int *dnumberofdynamicallyaddedvertices,
                                                int *ddynamicallyaddedvertices);

__global__ void DetectAndSetPendantPathsCase4(int nrVertices, 
                                                int k,
                                                int *match, 
                                                int *dforwardlinkedlist, 
                                                int *dbackwardlinkedlist, 
                                                int * dedgestatus,
                                                int *dlength, 
                                                int *dnumberofdynamicallyaddedvertices,
                                                int *ddynamicallyaddedvertices);                        

__global__ void FillSolutionArray(int leafIndex,
                                int * dsolution,
                                int sizeOfKernelSolution,
                                int2 * dbfssearchtree,
                                int * dnumberofdynamicallyaddedvertices,
                                int * ddynamicallyaddedvertices);

__global__ void PrintSolutionArray(int solutionSize,
                                int * dsolution);

__device__ void SetEdges(   int vertexToInclude,
                            int * dedgestatus);

__host__ __device__  int4 CalculateLeafOffsets(  int leafIndex,
                                                int fullpathcount);

class VCGPU
{

	public:
		VCGPU(const mtc::Graph &_graph, 
              const int &_threadsPerBlock, 
              const unsigned int &_barrier, 
              const unsigned int &_k,
              bool &solutionCantExist);
		~VCGPU();
        void CallDrawSearchTree(std::string prefix);

        long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);

        int4 numberCompletedPaths(int nrVertices, 
                                int leafIndex,
                                int depthOfLeaf,
                                int *dbackwardlinkedlist, 
                                int *dlength,
                                int recursiveStackDepth);

        int4 numberCompletedPathsTest(int nrVertices, 
                                int leafIndex,
                                int *dbackwardlinkedlist, 
                                int *dlength,
                                int recursiveStackDepth);


        int4 numberCompletedPathsTestP2(int nrVertices, 
                                int leafIndex,
                                int *dbackwardlinkedlist, 
                                int *dlength,
                                int recursiveStackDepth);

        void GetLengthStatistics(int nrVertices, 
                                int threadsPerBlock, 
                                int *dbackwardlinkedlist, 
                                int *dlength,
                                int *dreducedlength);
        
        void SetEdgesOfLeaf(int leafIndex);
        void Match(int leafIndex);
        void bussKernelizationP1();
        void bussKernelizationP2();
        void bussKernelizationP1(int root, int recursiveStackDepth, bool & foundSolution);
        void FindCover(int root, int recursiveStackDepth, bool & foundSolution);
        void EvaluateLeafBFS(int root, int recursiveStackDepth, bool & foundSolution);

        void PopulateBFSTree(int root, int recursiveStackDepth, bool & foundSolution);
        void ReinitializeArrays();
        void PrintData ();
        void CopyMatchingBackToHost(std::vector<int> & match);
        void GetDeviceVectors(int nrVertices, std::vector<int> & fll, std::vector<int> & bll, std::vector<int> & length);
        bool & solutionCantExist;
        long long sizeOfSearchTree;
        int k;
        int kPrime;
        int kPrimeBFS;
        int kPrimeDFS;

        int fullpathcount, depthOfSearchTree, remainingedges;
        std::vector<float> finishedLeavesPerLevel;
        std::vector<float> totalLeavesPerLevel;
        std::vector<int> edgestatus;
        std::vector<int> newdegrees;
        std::vector<int2> bfssearchtree;
        std::vector<int2> dfssearchtree;
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
        int2 *dbfssearchtree, *ddfssearchtree;
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
        thrust::device_vector<int> dmtch;
        thrust::device_vector<int> dfll;
        thrust::device_vector<int> dbll;

        thrust::host_vector<int> recursive_leaf_offsets;
        thrust::device_vector<int> d_recursive_leaf_offsets;

        thrust::host_vector<int> recursive_leaf_counters;
        thrust::device_vector<int> d_recursive_leaf_counters;

        mtc::GraphMatchingGeneralGPURandom matcher;

        GraphViz Gviz;
	protected:
		const mtc::Graph &graph;
        const int &threadsPerBlock;
        const unsigned int &barrier;
		//int2 *dneighbourRanges;
		//int *dneighbours;
        
};

#endif
