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
#include "vcgpu.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include "cub/cub.cuh"

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

texture<int2, cudaTextureType1D, cudaReadModeElementType> neighbourRangesTexture;
texture<int, cudaTextureType1D, cudaReadModeElementType> neighboursTexture;
texture<float, cudaTextureType1D, cudaReadModeElementType> weightsTexture;

VCGPU::VCGPU(const Graph &_graph, const int &_threadsPerBlock, const unsigned int &_barrier, const unsigned int &_k) :
		graph(_graph),
        threadsPerBlock(_threadsPerBlock),
        barrier(_barrier),
		matcher(_graph, _threadsPerBlock, _barrier),
        dfll(_graph.nrVertices),
        dbll(_graph.nrVertices),
        k(_k)
{
    sizeOfSearchTree = CalculateSpaceForDesiredNumberOfLevels(_k);
    // Wrong since numEdges < neighbors (up to double the num edges, in and out)
    //cudaMalloc(&dedgestatus, sizeof(int)*graph.nrEdges) != cudaSuccess || 
    if (cudaMalloc(&dedgestatus, sizeof(int)*graph.neighbours.size()) != cudaSuccess || 
        cudaMalloc(&dlength, sizeof(int)*graph.nrVertices) != cudaSuccess || 
        cudaMalloc(&dsearchtree, sizeof(int2)*sizeOfSearchTree) != cudaSuccess || 
        cudaMalloc(&dfullpathcount, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dnumleaves, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&active_frontier_status, sizeof(int)*sizeOfSearchTree) != cudaSuccess || 
        cudaMalloc(&ddegrees, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}
    ReinitializeArrays();
}

VCGPU::~VCGPU(){
    cudaFree(ddegrees);
	cudaFree(dlength);
    cudaFree(dsearchtree);
    cudaFree(dedgestatus);
    cudaFree(dfullpathcount);
    cudaFree(dnumleaves);
	cudaUnbindTexture(neighboursTexture);
	cudaUnbindTexture(neighbourRangesTexture);
}

__host__ __device__ long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i < NumberOfLevels; ++i)
        summand += pow (3.0, i);
    return summand;
}

void VCGPU::GetDeviceVectors(int nrVertices, std::vector<int> & fll, std::vector<int> & bll, std::vector<int> & length)
{
	//Copy obtained matching on the device back to the host.
	if (cudaMemcpy(&fll[0], dforwardlinkedlist, sizeof(int)*nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess ||
		cudaMemcpy(&bll[0], dbackwardlinkedlist, sizeof(int)*nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(&length[0], dlength, sizeof(int)*nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cerr << "Unable to retrieve data!" << endl;
		throw exception();
	}
}
void VCGPU::GetLengthStatistics(int nrVertices, int threadsPerBlock, int *dbackwardlinkedlist, int *dlength, int *dreducedlength)
{
	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
	ReducePathLengths<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, dbackwardlinkedlist, dlength, dreducedlength);
}

void VCGPU::numberCompletedPaths(int nrVertices, 
                        int *dbackwardlinkedlist, 
                        int *dlength){
	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    PopulateSearchTree<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, 
                                                            4,
                                                            dforwardlinkedlist,
                                                            dbackwardlinkedlist, 
                                                            dlength,
                                                            dfullpathcount,
                                                            dsearchtree);

    
    /*
    CalculateLeafOffsets<<<1, 1>>>(
                                    dfullpathcount,
                                    dnumleaves,
                                    active_frontier_status);
    */
}

// 2 Possibilities for recycling the paths of length 1&2
// Depending on whether we want to perform parallel frontier splitting.

// PFS (NO) - this is my first implementation.
// 1) Continue matching from a leaf, after removing edges 
// of included vertices and un-coloring the non-included vertices
// This approach will prioritize DF growth of the search tree.
// This allows us to only maintain 1 copy of the edge status in GPU mem.

// PFS (YES) - can try this in second implementation.
// 2) Evaluate each path for pendantness at each leaf node.
// This requires storing the edge status of each leaf node
// for any benefit to be seen.

//  However, it is MUCH easier to just wipe away intermediate paths
// And perform BFS at each leaf node, if we stick to BFS for as 
// long as complete levels can be formed.

// For DFS, we'd assume that every level beneath the last BFS has
// started at its left most child, and will need to be recursively
// searched from the bottom.

void VCGPU::FindCover(){
    std::vector<int> match;
    
    int leftMostLeafOfLevel = 0;
    int rightMostLeafOfLevel = 1;
    for (int activeRoot = leftMostLeafOfLevel; activeRoot < rightMostLeafOfLevel; ++activeRoot){
        matcher.initialMatching(match);
        ReinitializeArrays();
        SetEdgesOfLeaf(activeRoot);
        Match();
        // Need to pass device pointer to LOP
        numberCompletedPaths(graph.nrVertices, dbackwardlinkedlist, dlength);
		matcher.copyMatchingBackToHost(match);
    }
}


void VCGPU::SetEdgesOfLeaf(int leafIndex){
    // Root of search tree is empty.
    if (leafIndex == 0)
        return;
	int blocksPerGrid = ceil(logf(2*leafIndex + 1) / logf(3)) - (int)(leafIndex==0);
    SetEdges<<<blocksPerGrid, threadsPerBlock>>>(leafIndex,
                                                dedgestatus,
                                                ddegrees,
                                                dsearchtree);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}

void VCGPU::Match(){
    //Initialise timers.
    cudaEvent_t t0, t1, t2, t3;
    float time0, time1;

    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);
    
    cudaEventRecord(t0, 0);
    cudaEventSynchronize(t0);

    matcher.performMatching(&matcher.match[0], t1, t2, dforwardlinkedlist, dbackwardlinkedlist, dlength, ddegrees, dedgestatus);
    
    cudaEventElapsedTime(&time1, t1, t2);
    cudaEventRecord(t3, 0);
    cudaEventSynchronize(t3);
    //Measure the total elapsed time (including data transfer) and the calculation time.
    cudaEventElapsedTime(&time0, t0, t3);
    cudaEventElapsedTime(&time1, t1, t2);
    //Destroy timers.
    cudaEventDestroy(t3);
    cudaEventDestroy(t2);
    cudaEventDestroy(t1);
    cudaEventDestroy(t0);
}

void VCGPU::ReinitializeArrays(){
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dedgestatus),  1, size_t(graph.neighbours.size()));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dfullpathcount),  0, size_t(1));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dnumleaves),  0, size_t(1));
    // Only >= 0 are heads of full paths
    // Before implementing recursive backtracking, I can keep performing this memcpy to set degrees
    // and the remove tentative vertices to check a cover.
    cudaMemcpy(ddegrees, &graph.degrees[0], sizeof(int)*graph.nrVertices, cudaMemcpyHostToDevice);

	thrust::sequence(dfll.begin(),dfll.end());
	dforwardlinkedlist = thrust::raw_pointer_cast(&dfll[0]);
	
	thrust::sequence(dbll.begin(),dbll.end());
	dbackwardlinkedlist = thrust::raw_pointer_cast(&dbll[0]);
}


// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void PopulateSearchTree(int nrVertices, 
                                                int leafIndex,
                                                int *dforwardlinkedlist, 
                                                int *dbackwardlinkedlist, 
                                                int *dlength, 
                                                int *dfullpathcount,
                                                int2* dsearchtree){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	// If not a head to a path of length 4, return (leaving the headindex == -1)
    if (threadID >= nrVertices || 
        dlength[threadID] != 3 || 
        dbackwardlinkedlist[threadID] != threadID) 
            return;
    // Counter is incremented and old value is used to number full paths.
    int myPathIndex 

    int first = dforwardlinkedlist[threadID];
    int second = dforwardlinkedlist[first];
    int third = dforwardlinkedlist[second];
    int fourth = dforwardlinkedlist[third];

    int arbitraryParameter;

    int leavesToProcess = atomicAdd(&dfullpathcount[0], 1) + 1;
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for leavesToProcess < closed form
    // start from level 1, hence add a level if LTP > 0, 1 complete level 
    // Add 1 if LTP == 0 to prevent runtime error
    // LTP = 2
    // CL = 1
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
    int leavesFromIncompleteLevelLvl = powf(3.0, incompleteLevel) - (int)(leavesToProcess == 0); 
    int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, (incompleteLevel-1)+(int)(leavesToProcess != 0)))/(1.0 - 3.0) - (int)(leavesToProcess != 0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    arbitraryParameter = 3*((3*leafIndex)+1);
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, incompleteLevel-1) - 3)/6;
    int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex
    printf("Level Depth %d\n", incompleteLevel);
    printf("Level Width  %d\n", leavesFromIncompleteLevelLvl);
    printf("Size of Tree %d\n", treeSizeNotIncludingThisLevel);
    printf("Global level left offset (GLLO) %d\n", leftMostLeafIndexOfIncompleteLevel);
    printf("Displacement frok GLLO %d %d %d \n", levelOffset,
                                                levelOffset + 1,
                                                levelOffset + 2);

    // Test from root for now, this code can have an arbitrary root though
    dsearchtree[levelOffset + 0] = make_int2(first, third);
    dsearchtree[levelOffset + 1] = make_int2(second, third);
    dsearchtree[levelOffset + 2] = make_int2(second, fourth);
}



__global__ void ReducePathLengths(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dlength,
                            int* dreducedlength){}

__global__ void SetEdges(const int leafIndex,
                        int * dedgestatus,
                        int * ddegrees,
                        int2 *dsearchtree){

	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int numberOfLevelsToAscend = blockIdx.x;
    int i;
    //if (threadIdx.x == 0){
    int thisBlocksSearchTreeNode = leafIndex / pow (3.0, numberOfLevelsToAscend);
    //}

    int2 verticesInNode = dsearchtree[thisBlocksSearchTreeNode];
    if (threadIdx.x < blockDim.x/2){
        i = verticesInNode.x;
    } else {
        i = verticesInNode.y;
    }

    const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
    for (int j = indices.x + (threadIdx.x % blockDim.x/2); j < indices.y; j += blockDim.x/2){
        const int ni = tex1Dfetch(neighboursTexture, j);
        // Set out-edges
        ddegrees[ni] -= dedgestatus[ni];
        dedgestatus[ni] = 0;

        if (threadIdx.x == 0){
                ddegrees[ni] = 0;
        }
    }
    __syncthreads();
    // (u,v) is the form of edge pairs.  We are traversing over v's outgoing edges, 
    // looking for u as the destination and turning off that edge.
    bool foundChild, tmp;
    // There are two possibilities for parallelization here:
    // 1) Each thread will take an out edge, and then each thread will scan the edges leaving 
    // that vertex for the original vertex.
    //for (int edge = LB + threadIdx.x; edge < UB; edge += blockDim.x){

    // Basically, each thread is reading wildly different data
    // 2) 1 out edge is traversed at a time, and then all the threads scan
    // all the edges leaving that vertex for the original vertex.
    // This is the more favorable data access pattern.
    const int2 indices_curr = tex1Dfetch(neighbourRangesTexture, i);
    for (int j = indices_curr.x + (threadIdx.x % blockDim.x/2); j < indices_curr.y; j += blockDim.x/2){
        const int ni = tex1Dfetch(neighboursTexture, j);    
        const int2 indices_neighbor = tex1Dfetch(neighbourRangesTexture, ni);
          for (int j_n = indices_neighbor.x; j_n < indices_neighbor.y; ++j_n){
                const int nj = tex1Dfetch(neighboursTexture, j_n);       
                foundChild = i == nj;
                // Set in-edge
                // store edge status
                tmp = dedgestatus[nj];
                //   foundChild     tmp   (foundChild & tmp)  (foundChild & tmp)^tmp
                //1)      0          0            0                       0
                //2)      1          0            0                       0
                //3)      0          1            0                       1
                //4)      1          1            1                       0
                //
                // Case 1: isnt myChild and edge is off, stay off
                // Case 2: is myChild and edge is off, stay off
                // Case 3: isn't myChild and edge is on, stay on
                // Case 4: is myChild and edge is on, turn off
                // All this logic is necessary because we aren't using degree to set upperbound
                // we are using row offsets, which may include some edges turned off on a previous
                // pendant edge processing step.
                dedgestatus[nj] ^= (foundChild & tmp);
        }
    }
}

__global__ void InitDegrees(const int nrVertices,
                            int * ddegrees){
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= nrVertices) return;
    const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
    ddegrees[i] = indices.y - indices.x;
}

__global__ void CalculateNumberOfLeaves(int *dfullpathcount){

}


//int leafIndex = global_active_leaf_value[leafIndex];
// Solve recurrence relation 
// g(n) = 1/6*((2*C+3)*3^n - 3)
// C depends on leafIndex
// where g(0) = left-most child of depth 1
// where g(1) = left-most child of depth 2
// where g(2) = left-most child of depth 3
// ...
//int arbitraryParameter = 3*(3*leafIndex)+1);

// currently a single root is expanded in gpu memory at a time. 
// efforts were made in the FPT-kVC "done" branch to maintain multiple copies of the graph
// and explore the search tree in parallel.
__global__ void CalculateLeafOffsets(
                                        int * dfullpathcount,
                                        int * dnumleaves,
                                        int * active_frontier_status){
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int leafIndex;
    int arbitraryParameter;
    int leftMostLeafIndexOfFullLevel;
    int leftMostLeafIndexOfIncompleteLevel;
    #ifndef NDEBUG
    printf("globalIndex %d, CalculateLeafOffsets\n",globalIndex);
    printf("globalIndex %d, global_active_leaves_count_current %x\n",globalIndex, global_active_leaves_count_current[0]);
    #endif
    int leavesToProcess = dfullpathcount[globalIndex];
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for leavesToProcess < closed form
    // start from level 1, hence add a level if LTP > 0, 1 complete level 
    // Add 1 if LTP == 0 to prevent runtime error
    // LTP = 2
    // CL = 1
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    int completeLevel = floor(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
    // If LTP == 0, we dont want to create any new leaves
    // Therefore, we dont want to enter the for loops.
    // The active leaf writes itself as it's parent before the for loops
    // This is overwritten within the for loops if LTP > 0
    // CLL = 3
    int leavesFromCompleteLvl = powf(3.0, completeLevel) - (int)(leavesToProcess == 0);
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for closed form < leavesToProcess
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    // IL = 1
    int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3)) - (int)(leavesToProcess==0);
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Add 1 when leavesToProcess isn't 0, so we start counting from level 1
    // Also subtract the root, so we start counting from level 1
    // TSC = 3
    int treeSizeComplete = (1.0 - powf(3.0, completeLevel+(int)(leavesToProcess != 0)))/(1.0 - 3.0) - (int)(leavesToProcess != 0);
    // How many internal leaves to skip in complete level
    // RFC = 1
    int removeFromComplete = ((3*leavesToProcess - treeSizeComplete) + 3 - 1) / 3;
    // Leaves that are used in next level
    int leavesFromIncompleteLvl = 3*removeFromComplete;
    
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
    leafIndex = 0;
    arbitraryParameter = 3*((3*leafIndex)+1);
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    leftMostLeafIndexOfFullLevel = ((2*arbitraryParameter+3)*powf(3.0, completeLevel-1) - 3)/6;
    leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, incompleteLevel-1) - 3)/6;

    int totalNewActive = (leavesFromCompleteLvl - removeFromComplete) + leavesFromIncompleteLvl;
    printf("globalIndex %d, CalculateLeafOffsets\n",globalIndex);
    printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, completeLevel);
    printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, incompleteLevel);
    printf("Leaves %d, treeSizeComplete %d\n",leavesToProcess, treeSizeComplete);
    printf("Leaves %d, totalNewActive %d\n",leavesToProcess, totalNewActive);
    printf("Leaves %d, leavesFromCompleteLvl %d\n",leavesToProcess, leavesFromCompleteLvl);
    printf("Leaves %d, leavesFromIncompleteLvl %d\n",leavesToProcess, leavesFromIncompleteLvl);
    printf("Leaves %d, leftMostLeafIndexOfFullLevel %d\n",leavesToProcess, leftMostLeafIndexOfFullLevel);
    printf("Leaves %d, leftMostLeafIndexOfIncompleteLevel %d\n",leavesToProcess, leftMostLeafIndexOfIncompleteLevel);
    dnumleaves[0] = totalNewActive;
    /*
    active_leaf_offsets[0] = make_int4( leftMostLeafIndexOfFullLevel,
                                        leftMostLeafIndexOfFullLevel + leavesFromCompleteLvl,
                                        leftMostLeafIndexOfIncompleteLevel,
                                        leftMostLeafIndexOfIncompleteLevel + leavesFromIncompleteLvl);
    active_leaf_offsets[0] = leftMostLeafIndexOfFullLevel;
    active_leaf_offsets[1] = leftMostLeafIndexOfFullLevel + leavesFromCompleteLvl;
    active_leaf_offsets[2] = leftMostLeafIndexOfIncompleteLevel;
    active_leaf_offsets[3] = leftMostLeafIndexOfIncompleteLevel + leavesFromIncompleteLvl;
                                        */

}