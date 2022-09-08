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
#include "vcgpu.cuh"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include "cub/cub.cuh"

using namespace std;
using namespace mtc;

#include <iostream>


// RE?
#include <curses.h>
#include <ncurses.h>

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

VCGPU::VCGPU(const Graph &_graph, 
             const int &_threadsPerBlock, 
             const unsigned int &_barrier, 
             const unsigned int &_k,
             bool & _solutionCantExist) :
		graph(_graph),
        threadsPerBlock(_threadsPerBlock),
        barrier(_barrier),
		matcher(_graph, _threadsPerBlock, _barrier),
        dfll(_graph.nrVertices),
        dbll(_graph.nrVertices),
        k(_k),
        solutionCantExist(_solutionCantExist)
{
    solution.resize(k);
    if (cudaMalloc(&ddegrees, sizeof(int)*graph.nrVertices) != cudaSuccess || 
        cudaMalloc(&dsolution, sizeof(int)*k) != cudaSuccess || 
        cudaMalloc(&dremainingedges, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dsizeofkernelsolution, sizeof(int)*1))	
    {
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}
    bussKernelizationP1();
    if (sizeOfKernelSolution > k){
        printf("|S| = b (%d) > k (%d), no solution exists\n", sizeOfKernelSolution, k);
    } else {
        printf("|S| = b (%d) <= k (%d), a solution may exist\n", sizeOfKernelSolution, k);
    }
    solutionCantExist = sizeOfKernelSolution > k;
    kPrime = k - sizeOfKernelSolution;
    if(!solutionCantExist){
        printf("Setting k' = k %d - b %d = %d\n", k, sizeOfKernelSolution, kPrime);
        bussKernelizationP2();
        solutionCantExist = remainingedges > k*kPrime;
        if(remainingedges > k*kPrime){
            printf("|G'(E)| (%d) > k (%d) * k' (%d) = %d, no solution exists\n",remainingedges, k, kPrime, k*kPrime);
        } else {
            printf("|G'(E)| (%d) <= k (%d) * k' (%d) = %d, a solution may exist\n",remainingedges, k, kPrime, k*kPrime);
        }
    }
    depthOfSearchTree = kPrime/2;
    if (solutionCantExist){
        return;
    }
    finishedLeavesPerLevel.resize(depthOfSearchTree+1);
    totalLeavesPerLevel.resize(depthOfSearchTree+1);
    sizeOfSearchTree = CalculateSpaceForDesiredNumberOfLevels(depthOfSearchTree);
    printf("SIZE OF SEARCH TREE %lld\n", sizeOfSearchTree);
    searchtree.resize(sizeOfSearchTree);

    // Wrong since numEdges < neighbors (up to double the num edges, in and out)
    //cudaMalloc(&dedgestatus, sizeof(int)*graph.nrEdges) != cudaSuccess || 
    if (cudaMalloc(&dedgestatus, sizeof(int)*graph.neighbours.size()) != cudaSuccess || 
        cudaMalloc(&dedges, sizeof(mtc::Edge)*graph.nrEdges) != cudaSuccess || 
        cudaMalloc(&dlength, sizeof(int)*graph.nrVertices) != cudaSuccess || 
        cudaMalloc(&dsearchtree, sizeof(int2)*sizeOfSearchTree) != cudaSuccess || 
        cudaMalloc(&duncoverededges, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dfullpathcount, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dnumleaves, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dremainingedges, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&dnumberofdynamicallyaddedvertices, sizeof(int)*1) != cudaSuccess || 
        cudaMalloc(&ddynamicallyaddedvertices_csr, sizeof(int)*(depthOfSearchTree+1)) != cudaSuccess || 
        cudaMalloc(&ddynamicallyaddedvertices, sizeof(int)*(kPrime)) != cudaSuccess ||        
        cudaMalloc(&dfinishedLeavesPerLevel, sizeof(float)*(depthOfSearchTree+1)) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}
    dmtch.resize(graph.nrVertices);
    dmatch = thrust::raw_pointer_cast(&dmtch[0]);

    edgestatus.resize(graph.neighbours.size());
    newdegrees.resize(graph.nrVertices);
    dynamcverts.resize(graph.nrVertices);

    cudaMemcpy(dedges, &graph.edges[0], sizeof(mtc::Edge)*graph.nrEdges, cudaMemcpyHostToDevice);

    // Since these are 32 byte sets, simply double for int2
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dsearchtree),  0, size_t(2*sizeOfSearchTree));

    ReinitializeArrays();
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	cudaBindTexture(0, neighbourRangesTexture, (void *)matcher.dneighbourRanges, neighbourRangesTextureDesc, sizeof(int2)*graph.neighbourRanges.size());
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	cudaBindTexture(0, neighboursTexture, (void *)matcher.dneighbours, neighboursTextureDesc, sizeof(int)*graph.neighbours.size());


}

VCGPU::~VCGPU(){
    if(solutionCantExist){
        cudaFree(ddegrees);
        cudaFree(dsolution);
        cudaFree(dremainingedges);
        cudaFree(dsizeofkernelsolution);
    } else {
        cudaFree(dedgestatus);
        cudaFree(dedges);
        cudaFree(dlength);
        cudaFree(dsearchtree);
        cudaFree(duncoverededges);
        cudaFree(dfullpathcount);
        cudaFree(dnumleaves);
        cudaFree(dremainingedges);
        cudaFree(dnumberofdynamicallyaddedvertices);
        cudaFree(ddynamicallyaddedvertices_csr);
        cudaFree(ddynamicallyaddedvertices);
        cudaFree(dfinishedLeavesPerLevel);
        cudaFree(dsolution);
        cudaFree(ddegrees);
        printf("Finished deallocating memory\n");
        cudaUnbindTexture(neighboursTexture);
        cudaUnbindTexture(neighbourRangesTexture);
    }
}

long long VCGPU::CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i <= NumberOfLevels; ++i){
        summand += pow (3.0, i);
        finishedLeavesPerLevel[i] = i ? 0 : 1;
        totalLeavesPerLevel[i] = pow (3.0, i);
    }
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

int4 VCGPU::numberCompletedPaths(int nrVertices, 
                        int leafIndex,
                        int depthOfLeaf,
                        int *dbackwardlinkedlist, 
                        int *dlength,
                        int recursiveStackDepth){
	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    PopulateSearchTree<<<blocksPerGrid, threadsPerBlock>>>(nrVertices,
                                                            sizeOfSearchTree, 
                                                            depthOfSearchTree,
                                                            leafIndex,
                                                            dfinishedLeavesPerLevel,
                                                            dforwardlinkedlist,
                                                            dbackwardlinkedlist, 
                                                            dlength,
                                                            dfullpathcount,
                                                            dsearchtree);
    // Dont bother with looking for pendants if I'm out of space.
    if (numberofdynamicallyaddedvertices<kPrime){
        DetectAndSetPendantPathsCase3<<<blocksPerGrid, threadsPerBlock>>>(nrVertices,
                                                            kPrime,
                                                            dmatch,
                                                            dforwardlinkedlist,
                                                            dbackwardlinkedlist,
                                                            dedgestatus, 
                                                            dlength,
                                                            dnumberofdynamicallyaddedvertices,
                                                            ddynamicallyaddedvertices);
        DetectAndSetPendantPathsCase4<<<blocksPerGrid, threadsPerBlock>>>(nrVertices,
                                                            kPrime,
                                                            dmatch,
                                                            dforwardlinkedlist,
                                                            dbackwardlinkedlist,
                                                            dedgestatus, 
                                                            dlength,
                                                            dnumberofdynamicallyaddedvertices,
                                                            ddynamicallyaddedvertices);
    }
    // Create CSR entry for dynamically added verts
    cudaMemcpy(&ddynamicallyaddedvertices_csr[recursiveStackDepth], dnumberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToDevice);


    cudaMemcpy(&fullpathcount, &dfullpathcount[0], sizeof(int)*1, cudaMemcpyDeviceToHost);
    //cudaMemcpy(&numberofdynamicallyaddedvertices, &dnumberofdynamicallyaddedvertices[0], sizeof(int)*1, cudaMemcpyDeviceToHost);

    int4 myActiveLeaves = CalculateLeafOffsets(leafIndex,
                                                fullpathcount);


    //printf("My active leaves %d %d %d %d\n", myActiveLeaves.x, myActiveLeaves.y, myActiveLeaves.z, myActiveLeaves.w);
    return myActiveLeaves;
}

int4 VCGPU::numberCompletedPathsTest(int nrVertices, 
                        int leafIndex,
                        int *dbackwardlinkedlist, 
                        int *dlength,
                        int recursiveStackDepth){

    fullpathcount = 1+rand()%20;

    //cudaMemcpy(&dfullpathcount[0], &fullpathcount, sizeof(int)*1, cudaMemcpyHostToDevice);

	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    PopulateSearchTreeTest<<<blocksPerGrid, threadsPerBlock>>>(nrVertices,
                                                            sizeOfSearchTree, 
                                                            depthOfSearchTree,
                                                            leafIndex,
                                                            dfinishedLeavesPerLevel,
                                                            dforwardlinkedlist,
                                                            dbackwardlinkedlist, 
                                                            dlength,
                                                            dfullpathcount,
                                                            dsearchtree,
                                                            fullpathcount);

    cudaMemcpy(&fullpathcount, &dfullpathcount[0], sizeof(int)*1, cudaMemcpyDeviceToHost);
    //cudaMemcpy(&numberofdynamicallyaddedvertices, &dnumberofdynamicallyaddedvertices[0], sizeof(int)*1, cudaMemcpyDeviceToHost);

    int4 myActiveLeaves = CalculateLeafOffsets(leafIndex,
                                                fullpathcount);


    //printf("My active leaves %d %d %d %d\n", myActiveLeaves.x, myActiveLeaves.y, myActiveLeaves.z, myActiveLeaves.w);
    return myActiveLeaves;
}

// Initial kernelization before search tree is built
void VCGPU::bussKernelizationP1(){
    cudaMemcpy(ddegrees, graph.degrees.data(), sizeof(int)*graph.nrVertices, cudaMemcpyHostToDevice);
    int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    BussKernelizationP1Kernel<<<blocksPerGrid, threadsPerBlock>>>(graph.nrVertices, 
                                                                k, 
                                                                ddegrees,
                                                                dsolution,
                                                                dsizeofkernelsolution);
    cudaMemcpy(&sizeOfKernelSolution, dsizeofkernelsolution, sizeof(int)*1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&solution[0], dsolution, sizeof(int)*sizeOfKernelSolution, cudaMemcpyDeviceToHost);
}

// Initial kernelization before search tree is built
void VCGPU::bussKernelizationP2(){
    printf("Remaining edges before Kernel %d\n", graph.nrEdges);
    // Using the indices to calculate degrees requires doubling and then halving
    // Since each edge is counted twice, once in each connecting vertex's indices.x to indices.y
    remainingedges = 2*graph.nrEdges;
    cudaMemcpy(dremainingedges, &remainingedges, sizeof(int)*1, cudaMemcpyHostToDevice);
    int blocksPerGrid = (sizeOfKernelSolution + threadsPerBlock - 1)/threadsPerBlock;
    //printf("Launching %d blocks for a solution of size %d\n", blocksPerGrid, sizeOfKernelSolution);
    BussKernelizationP2Kernel<<<blocksPerGrid, threadsPerBlock>>>(sizeOfKernelSolution,
                                                                ddegrees,
                                                                dremainingedges,
                                                                dsolution);
    cudaMemcpy(&remainingedges, dremainingedges, sizeof(int)*1, cudaMemcpyDeviceToHost);
    // Using the indices to calculate degrees requires doubling and then halving
    // Since each edge is counted twice, once in each connecting vertex's indices.x to indices.y
    remainingedges/=2;
    printf("Remaining edges after Kernel %d\n", remainingedges);
}

void VCGPU::bussKernelizationP1(int root, int recursiveStackDepth, bool & foundSolution){

}


/*
void VCGPU::eraseDynVertsOfRecursionLevel(int recursiveStackDepth){

    cudaMemcpy(&numberofdynamicallyaddedverticesLB, ddynamicallyaddedvertices_csr[recursiveStackDepth], sizeof(int)*1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&numberofdynamicallyaddedverticesUB, ddynamicallyaddedvertices_csr[recursiveStackDepth+1], sizeof(int)*1, cudaMemcpyDeviceToHost);
    int numberToErase = numberofdynamicallyaddedverticesUB - numberofdynamicallyaddedverticesLB;

    cudaMemcpy(&numberofdynamicallyaddedvertices, dnumberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToHost);
    numberofdynamicallyaddedvertices -= numberToErase;
    cudaMemcpy(&dnumberofdynamicallyaddedvertices, numberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToHost);

}
*/
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

// Will most likely copy back the frontier bool array and iterate through each frontier one at a time
// for v1.
//for (int activeRoot = leftMostLeafOfLevel; activeRoot < rightMostLeafOfLevel; ++activeRoot){

// For DFS, we'd assume that every level beneath the last BFS has
// started at its left most child, and will need to be recursively
// searched from the bottom.

// It's important that this method is called for the leaf
// nodes, but that number of new leaves == 0
// so the pendant paths are found if any remain.
void VCGPU::FindCover(int root,
                      int recursiveStackDepth,
                      bool & foundSolution){
    if (foundSolution)
        return;

	int blocksPerGrid = (graph.nrEdges + threadsPerBlock - 1)/threadsPerBlock;
    int depthOfLeaf = ceil(logf(2*root + 1) / logf(3)) - 1;

    /*
    int depthOfLeaf;
    if (root)
        depthOfLeaf = ceil(logf(2*root + 1) / logf(3)) - 1;
    else
        depthOfLeaf = root;
    */

    #ifndef NDEBUG
    printf("Called FindCover li %d rl %d \n", root, recursiveStackDepth);
    printf("depthOfLeaf %d depthOfSearchTree %d\n",  depthOfLeaf, depthOfSearchTree);
    #endif    
    cudaMemcpy(&finishedLeavesPerLevel[1], &dfinishedLeavesPerLevel[1], sizeof(float)*depthOfSearchTree, cudaMemcpyDeviceToHost);

    curs_set (0);
    for(int i = 0; i <= depthOfSearchTree; ++i){
        mvprintw (i, 4, "Depth %d %f Complete %f/%f\n", i, finishedLeavesPerLevel[i]/totalLeavesPerLevel[i], finishedLeavesPerLevel[i], totalLeavesPerLevel[i]);
    }
    refresh ();
    if (depthOfLeaf > depthOfSearchTree){
        return;
    }

    // If you want to check the quality of each match, uncomment
    // Else, the only noticable changes will be in the recursion stack 
    // and the device search tree.
    //std::vector<int> match;
    //matcher.initialMatching(match);

//    printf("\033[A\33[2K\rCalling Find Cover from %d, level depth of leaf %d\n", root, depthOfLeaf);
    numoftreeverts = 2*(depthOfLeaf);
    if (sizeOfKernelSolution+numoftreeverts+numofdynamcverts < k) {
        ReinitializeArrays();
        cudaDeviceSynchronize();
        // TODO - Need to set the pendant vertices also.
        SetEdgesOfLeaf(root);

        // Test algebra, comment Match(root)
        Match(root);
        cudaDeviceSynchronize();

        //matcher.copyMatchingBackToHost(match);
        // Need to pass device pointer to LOP
        // Test algebra, use Test
        // Might have an error if 1 single path found.
        cudaMemcpy(&numberofdynamicallyaddedvertices, &dnumberofdynamicallyaddedvertices[0], sizeof(int)*1, cudaMemcpyDeviceToHost);
        int4 newLeaves = numberCompletedPaths(graph.nrVertices, root, depthOfLeaf, dbackwardlinkedlist, dlength, recursiveStackDepth);
        cudaMemcpy(&numberofdynamicallyaddedvertices, &dnumberofdynamicallyaddedvertices[0], sizeof(int)*1, cudaMemcpyDeviceToHost);

        //int4 newLeaves = numberCompletedPathsTest(graph.nrVertices, root, dbackwardlinkedlist, dlength, recursiveStackDepth);
        cudaDeviceSynchronize();
        cudaMemcpy(&finishedLeavesPerLevel[1], &dfinishedLeavesPerLevel[1], sizeof(float)*(depthOfSearchTree), cudaMemcpyDeviceToHost);

        curs_set (0);
        for(int i = 0; i <= depthOfSearchTree; ++i){
            mvprintw (i, 4, "Depth %d %f Complete %f/%f\n", i, finishedLeavesPerLevel[i]/totalLeavesPerLevel[i], finishedLeavesPerLevel[i], totalLeavesPerLevel[i]);
        }
        refresh ();
        if (root != 0){
            cuMemsetD32(reinterpret_cast<CUdeviceptr>(duncoverededges),  0, size_t(1));
            cudaDeviceSynchronize();
            EvaluateSingleLeafNode<<<blocksPerGrid, threadsPerBlock, (sizeOfKernelSolution + 2*(depthOfLeaf+1))*sizeof(int)>>>(
                                                                                                graph.nrEdges,
                                                                                                root,
                                                                                                depthOfLeaf,
                                                                                                sizeOfKernelSolution,
                                                                                                dsolution,
                                                                                                dedges, 
                                                                                                dsearchtree,
                                                                                                dnumberofdynamicallyaddedvertices,
                                                                                                ddynamicallyaddedvertices,
                                                                                                duncoverededges);
            cudaMemcpy(&uncoverededges, duncoverededges, sizeof(int)*1, cudaMemcpyDeviceToHost);
            if (uncoverededges){
                #ifndef NDEBUG
                printf("leaf index %d is not a solution :  %d edges are uncovered\n", root, uncoverededges);
                #endif
            } else {   
                printf("leaf index %d is a solution :  %d edges are uncovered\n", root, uncoverededges);
                FillSolutionArray<<<1,1>>>(root,
                                    dsolution,
                                    sizeOfKernelSolution,
                                    dsearchtree,
                                    dnumberofdynamicallyaddedvertices,
                                    ddynamicallyaddedvertices);

                cudaMemcpy(&numofdynamcverts, dnumberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToHost);
                

                numoftreeverts = 2*(depthOfLeaf);
                printf("sizeOfKernelSolution %d numoftreeverts %d numofdynamcverts %d\n", sizeOfKernelSolution,numoftreeverts,numofdynamcverts);
                solutionSize = sizeOfKernelSolution+numoftreeverts+numofdynamcverts;
                cudaMemcpy(solution.data(), dsolution, sizeof(int)*solutionSize, cudaMemcpyDeviceToHost);

                foundSolution = true;
                cudaDeviceSynchronize();
                checkLastErrorCUDA(__FILE__, __LINE__);    
                cudaMemcpy(&remainingedges, dremainingedges, sizeof(int)*1, cudaMemcpyDeviceToHost);
                cudaMemcpy(&edgestatus[0], dedgestatus, sizeof(int)*graph.neighbours.size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(&newdegrees[0], ddegrees, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost);
                cudaMemcpy(&searchtree[0], dsearchtree, sizeof(int2)*searchtree.size(), cudaMemcpyDeviceToHost);
                cudaMemcpy(&numofdynamcverts, dnumberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToHost);
                cudaMemcpy(&dynamcverts[0], ddynamicallyaddedvertices, sizeof(int)*numofdynamcverts, cudaMemcpyDeviceToHost);

                Gviz.DrawInputGraphColored(graph,
                                        root,
                                        searchtree,
                                        numofdynamcverts,
                                        dynamcverts,
                                        dmtch,
                                        dfll,
                                        dbll,
                                        root);
                Gviz.DrawSearchTree(sizeOfSearchTree,
                                &searchtree[0],
                                root);  

                return;
            }
        }

        #ifndef NDEBUG
        cudaMemcpy(&remainingedges, dremainingedges, sizeof(int)*1, cudaMemcpyDeviceToHost);
        cudaMemcpy(&edgestatus[0], dedgestatus, sizeof(int)*graph.neighbours.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(&newdegrees[0], ddegrees, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost);
        cudaMemcpy(&searchtree[0], dsearchtree, sizeof(int2)*searchtree.size(), cudaMemcpyDeviceToHost);
        cudaMemcpy(&numofdynamcverts, dnumberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToHost);
        cudaMemcpy(&dynamcverts[0], ddynamicallyaddedvertices, sizeof(int)*numofdynamcverts, cudaMemcpyDeviceToHost);

        Gviz.DrawInputGraphColored(graph,
                                root,
                                searchtree,
                                numofdynamcverts,
                                dynamcverts,
                                dmtch,
                                dfll,
                                dbll,
                                root);
        Gviz.DrawSearchTree(sizeOfSearchTree,
                        &searchtree[0],
                        root);   
            
        printf("leaf index %d LO %d %d LO %d %d\n", root, newLeaves.x, newLeaves.y, newLeaves.z, newLeaves.w);
        #endif
        while(newLeaves.x < newLeaves.y && newLeaves.x < sizeOfSearchTree){
            FindCover(newLeaves.x, recursiveStackDepth+1, foundSolution);
            ++newLeaves.x;
        }
        depthOfLeaf = ceil(logf(2*newLeaves.z + 1) / logf(3)) - 1;
        while(newLeaves.z < newLeaves.w && newLeaves.z < sizeOfSearchTree){
            FindCover(newLeaves.z, recursiveStackDepth+1, foundSolution);
            ++newLeaves.z;
        }
    }
    //PrintData (); 
    // Wipe away my pendant nodes from shared list
    eraseDynVertsOfRecursionLevel<<<1, threadsPerBlock>>>(recursiveStackDepth,
                                              dnumberofdynamicallyaddedvertices, 
                                              ddynamicallyaddedvertices_csr, 
                                              ddynamicallyaddedvertices);
    cudaMemcpy(&numofdynamcverts, dnumberofdynamicallyaddedvertices, sizeof(int)*1, cudaMemcpyDeviceToHost);

}

void VCGPU::CallDrawSearchTree(std::string prefix){
    cudaMemcpy(&searchtree[0], dsearchtree, sizeof(int2)*searchtree.size(), cudaMemcpyDeviceToHost);
    Gviz.DrawSearchTree(sizeOfSearchTree,
					&searchtree[0],
					prefix); 
}



void VCGPU::SetEdgesOfLeaf(int leafIndex){
    // Root of search tree is empty.
    if (leafIndex == 0)
        return;
	int blocksPerGrid = 2*(ceil(logf(2*leafIndex + 1) / logf(3)) - (int)(leafIndex==0));
    SetEdges<<<blocksPerGrid, threadsPerBlock>>>(leafIndex,
                                                dedgestatus,
                                                ddegrees,
                                                dsearchtree);
	blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    CalculateDegrees<<<blocksPerGrid, threadsPerBlock>>>(graph.nrVertices,
                                                dedgestatus,
                                                ddegrees);
    cudaDeviceSynchronize();
    checkLastErrorCUDA(__FILE__, __LINE__);
}

void VCGPU::PrintData (){
   
    printf("neighbours size %d num edges %d\n",  graph.neighbours.size(), graph.nrEdges);
    printf("Row Offs\n");
    for (int i = 0; i < graph.nrVertices; ++i){
        printf("%d %d, ",graph.neighbourRanges[i].x, graph.neighbourRanges[i].y);
    }
    printf("\n");
    printf("Cols\n");
    for (int i = 0; i < graph.neighbours.size(); ++i){
        printf("%d ",graph.neighbours[i]);
    }
    printf("\n");
    printf("Vals\n");
    for (int i = 0; i < graph.neighbours.size(); ++i){
        printf("%d ",edgestatus[i]);
    }
    printf("\n");
    printf("Degrees\n");
    for (int i = 0; i < graph.nrVertices+1; ++i){
        printf("%d ", newdegrees[i]);
    }
    printf("Remaining Edges\n");
    printf("%d\n", remainingedges);
    printf("\n");
    printf("Dyn Verts\n");
    for (int i = 0; i < numofdynamcverts; ++i){
        printf("%d ", dynamcverts[i]);
    }

}
void VCGPU::Match(int leafIndex){
    //Initialise timers.
    cudaEvent_t t0, t1, t2, t3;
    float time0, time1;

    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);
    
    cudaEventRecord(t0, 0);
    cudaEventSynchronize(t0);

    matcher.performMatching(dmatch, t1, t2, dforwardlinkedlist, dbackwardlinkedlist, dlength, dsearchtree, ddynamicallyaddedvertices, dnumberofdynamicallyaddedvertices, sizeOfKernelSolution, leafIndex);
    
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
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dlength),  0, size_t(graph.nrVertices));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dfullpathcount),  0, size_t(1));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dnumleaves),  0, size_t(1));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dremainingedges),  0, size_t(1));
    // Only >= 0 are heads of full paths
    // Before implementing recursive backtracking, I can keep performing this memcpy to set degrees
    // and the remove tentative vertices to check a cover.
    cudaMemcpy(ddegrees, &graph.degrees[0], sizeof(int)*graph.nrVertices, cudaMemcpyHostToDevice);

    dfll.clear();
    dfll.resize(graph.nrVertices);
	thrust::sequence(dfll.begin(),dfll.end(), 0, 1);
	dforwardlinkedlist = thrust::raw_pointer_cast(&dfll[0]);
	
    dbll.clear();
    dbll.resize(graph.nrVertices);
	thrust::sequence(dbll.begin(),dbll.end(), 0, 1);
	dbackwardlinkedlist = thrust::raw_pointer_cast(&dbll[0]);

}

__global__ void ReduceEdgeStatusArray(int nrNeighbors,
							int *dedgestatus,
                            int* dremainingedges){
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
    int idx = threadIdx.x+blockDim.x*blockIdx.x;
    int val = 0;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;
    while (idx < nrNeighbors) { // grid stride loop to load
        val += dedgestatus[idx];
        idx += gridDim.x*blockDim.x;
    }
    // 1st warp-shuffle reduction
    for (int offset = warpSize/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
    if (lane == 0) sdata[warpID] = val;
    __syncthreads(); // put warp results in shared mem
    // hereafter, just warp 0
    if (warpID == 0){
        // reload val from shared mem if warp existed
        val = (tid < blockDim.x/warpSize)?sdata[lane]:0;
        // final warp-shuffle reduction
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(mask, val, offset);
        if (tid == 0) atomicAdd(dremainingedges, val);
    }
}

// Only launch 1 block so sync threads prevents retrieving a bad UB
__global__ void eraseDynVertsOfRecursionLevel(int recursiveStackDepth,
                                              int * dnumberofdynamicallyaddedvertices, 
                                              int * ddynamicallyaddedvertices_csr, 
                                              int * ddynamicallyaddedvertices){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    int LB = ddynamicallyaddedvertices_csr[recursiveStackDepth];
    int UB = ddynamicallyaddedvertices_csr[recursiveStackDepth+1];
    for (int entry = LB + threadID; entry < UB; entry += blockDim.x)
        ddynamicallyaddedvertices[entry] = 0;

    __syncthreads();
    if (threadID == 0){
        dnumberofdynamicallyaddedvertices -= (UB - LB);
        ddynamicallyaddedvertices_csr[recursiveStackDepth+1] = 0;
    }
}

// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void PopulateSearchTree(int nrVertices, 
                                    int sizeOfSearchTree,
                                    int depthOfSearchTree,
                                    int leafIndex,
                                    float * dfinishedLeavesPerLevel,
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

    int first = threadID;
    int second = dforwardlinkedlist[first];
    int third = dforwardlinkedlist[second];
    int fourth = dforwardlinkedlist[third];

    int leavesToProcess = atomicAdd(&dfullpathcount[0], 1) + 1;
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // r = 3, a = 1, solve for n given s_n = leavesToProcess ∈ [1,m/4]
    // where m = number of vertices.
    // s_n = (1-r^(n+1))/(1-r)
    // s_n * (1-3) = -2*s_n = (1-r^(n+1))
    //     = -2*s_n - 1 = -3^(n+1)
    //     =  2*s_n + 1  =  3^(n+1)
    //     =  log(2*s_n + 1) = n+1*log(3)
    //     =  log(2*s_n + 1)/log(3) = n + 1
    //     =  log(2*s_n + 1)/log(3) - 1 = n
    // n is the number of terms in the closed form solution.
    // Alternatively, n is the number of levels in the search tree.
    int n = ceil(logf(2*leavesToProcess + 1) / logf(3));
    //float nf = logf(2*leavesToProcess + 1) / logf(3);
    int arbitraryParameter = 3*((3*leafIndex)+1);
    // At high powers, the error of transendental powf causes bugs.
    //int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

    // Discrete calculation without trancendentals
    int leftMostLeafIndexOfIncompleteLevel = (2*arbitraryParameter+3);
    int multiplicand = 1;
    for (int i = 1; i < n; ++i)
        multiplicand*=3;
    leftMostLeafIndexOfIncompleteLevel*=multiplicand;
    leftMostLeafIndexOfIncompleteLevel-=3;
    leftMostLeafIndexOfIncompleteLevel/=6;

    int treeSizeNotIncludingThisLevel = (1.0 - multiplicand)/(1.0 - 3.0); 
    // At high powers, the error of transendental powf causes bugs. 
    //int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, ((n+1)-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons???
    int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    //int internalLeafIndex = leavesToProcess - treeSizeNotIncludingThisLevel;
    int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

    if (sizeOfSearchTree <= levelOffset ||
        sizeOfSearchTree <= (levelOffset + 1) ||
        sizeOfSearchTree <= (levelOffset +2)||
        levelOffset< 0 ||
        (levelOffset+1) < 0 ||
        (levelOffset + 2)< 0){
            atomicSub(&dfullpathcount[0], 1);
            //printf("child %d exceeded srch tree depth\n", levelOffset);
            return;
    }
    // Add to device pointer of level
    int depthOfLeaf = floor(logf(2*levelOffset + 1) / logf(3));
    //printf("leafIndex %d atomicAdd(&dfinishedLeavesPerLevel[%d], 3) newleaves %d - %d\n", leafIndex, depthOfLeaf,levelOffset, levelOffset + 2); 
    atomicAdd(&dfinishedLeavesPerLevel[depthOfLeaf], 3.0); 
    // Test from root for now, this code can have an arbitrary root though
    dsearchtree[levelOffset + 0] = make_int2(first, third);
    dsearchtree[levelOffset + 1] = make_int2(second, third);
    dsearchtree[levelOffset + 2] = make_int2(second, fourth);   
}


// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void PopulateSearchTreeTest(int nrVertices, 
                                    int sizeOfSearchTree,
                                    int depthOfSearchTree,
                                    int leafIndex,
                                    float * dfinishedLeavesPerLevel,
                                    int *dforwardlinkedlist, 
                                    int *dbackwardlinkedlist, 
                                    int *dlength, 
                                    int *dfullpathcount,
                                    int2* dsearchtree,
                                    int fullpathcount){
	const unsigned int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	// If not a head to a path of length 4, return (leaving the headindex == -1)
    if (threadID >= fullpathcount) 
            return;

    unsigned int leavesToProcess = atomicAdd(&dfullpathcount[0], 1) + 1;
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // r = 3, a = 1, solve for n given s_n = leavesToProcess ∈ [1,m/4]
    // where m = number of vertices.
    // s_n = (1-r^(n+1))/(1-r)
    // s_n * (1-3) = -2*s_n = (1-r^(n+1))
    //     = -2*s_n - 1 = -3^(n+1)
    //     = 2*s_n + 1  =  3^(n+1)
    //     = log(2*s_n + 1) = n+1*log(3)
    //     = log(2*s_n + 1)/log(3) = n + 1
    //     = log(2*s_n + 1)/log(3) - 1 = n
    // n is the number of terms in the closed form solution.
    // Alternatively, n is the number of levels in the search tree.
    unsigned int n = ceil(logf(2*leavesToProcess + 1) / logf(3));
    //float nf = logf(2*leavesToProcess + 1) / logf(3);
    unsigned int arbitraryParameter = 3*((3*leafIndex)+1);
    // At high powers, the error of transendental powf causes bugs.
    //unsigned int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

    // Discrete calculation without trancendentals
    unsigned int leftMostLeafIndexOfIncompleteLevel = (2*arbitraryParameter+3);
    unsigned int multiplicand = 1;
    for (unsigned int i = 1; i < n; ++i)
        multiplicand*=3;
    leftMostLeafIndexOfIncompleteLevel*=multiplicand;
    leftMostLeafIndexOfIncompleteLevel-=3;
    leftMostLeafIndexOfIncompleteLevel/=6;

    unsigned int treeSizeNotIncludingThisLevel = (1.0 - multiplicand)/(1.0 - 3.0); 
    // At high powers, the error of transendental powf causes bugs. 
    //unsigned int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, ((n+1)-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons???
    unsigned int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    //unsigned int internalLeafIndex = leavesToProcess - treeSizeNotIncludingThisLevel;
    unsigned int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

    if (sizeOfSearchTree <= levelOffset ||
        sizeOfSearchTree <= (levelOffset + 1) ||
        sizeOfSearchTree <= (levelOffset +2)||
        levelOffset< 0 ||
        (levelOffset+1) < 0 ||
        (levelOffset + 2)< 0){
            atomicSub(&dfullpathcount[0], 1);
            //printf("child %d exceeded srch tree depth\n", levelOffset);
            return;
    }
    // Add to device pointer of level
    //atomicAdd(&dfinishedLeavesPerLevel[depthOfLeaf], 3); 
    // Test from root for now, this code can have an arbitrary root though
    dsearchtree[levelOffset + 0] = make_int2(levelOffset, levelOffset);
    dsearchtree[levelOffset + 1] = make_int2(levelOffset + 1, levelOffset + 1);
    dsearchtree[levelOffset + 2] = make_int2(levelOffset + 2, levelOffset + 2);   
}

// Each thread will take an edge.  Each thread will loop through the answer
// until it finds either vertex a or b of an edge.
// if it reaches the end of the answer without terminating, it isn't a solution.
// Amount of shared memory should be 2*depth*sizeof(unsigned int)
// 
__global__ void EvaluateSingleLeafNode(int nrEdges,
                                    int leafIndex,
                                    int depthOfLeaf,
                                    int sizeOfKernelSolution,
                                    int * dsolution,
                                    mtc::Edge * dedges, 
                                    int2 * dsearchtree,
                                    int * dnumberofdynamicallyaddedvertices,
                                    int * ddynamicallyaddedvertices,
                                    int * uncoverededges){
    extern __shared__ int soln[];
	const int edgeID = blockIdx.x*blockDim.x + threadIdx.x;
    if (edgeID >= nrEdges)
        return;
    int UBDyn = dnumberofdynamicallyaddedvertices[0];
    const int tid = threadIdx.x;
    #ifndef NDEBUG
    if (tid == 0){
        printf("depthOfLeaf %d\n", depthOfLeaf);
        printf("UBDyn %d\n", UBDyn);
    }
    #endif
    int leafIndexSoln = leafIndex;
    int2 nodeEntry;
    int counter = 0;
    int i;
    // Load solution into shared memory
    for(i = threadIdx.x; i < sizeOfKernelSolution; i+=blockDim.x){
        soln[i] = dsolution[i];
    }
    // Currently the same first sizeOfKernelSolution threads
    // have to first copy the kernel solution
    // then copy the tree solution.
    // This can occur in parallel with fancy for loop indexing.
    if (tid <= depthOfLeaf){
        for(int i = 0; i < tid; ++i){
            if(leafIndexSoln % 3 == 0){
                --leafIndexSoln;
                leafIndexSoln = leafIndexSoln / 3;
            } else {
                leafIndexSoln = leafIndexSoln / 3;
            }
        }
        nodeEntry = dsearchtree[leafIndexSoln];
        soln[sizeOfKernelSolution + 2*tid] = nodeEntry.x;
        soln[sizeOfKernelSolution + 2*tid + 1] = nodeEntry.y;
        #ifndef NDEBUG
        printf("tid %d is adding %d %d to soln indices %d %d \n", tid, nodeEntry.x, nodeEntry.y, 2*tid, 2*tid + 1);
        #endif
    }
        /*
        // If you decide to use sm for dynamic verts
        for (int index = 0; index < dnumberofdynamicallyaddedvertices[0]; ++index){
            soln[counter] = ddynamicallyaddedvertices[index];
            ++counter;
        }
        */
     
    __syncthreads(); // put warp results in shared mem
    #ifndef NDEBUG
    if (threadIdx.x == 0){
        printf("LI %d Tree soln\n", leafIndex);
        for (int i = 0; i < (depthOfLeaf+1)*2; ++i)
            printf("%d ", soln[i]);
        printf("\n");
    }
    __syncthreads(); // put warp results in shared mem
    #endif
    Edge & edge = dedges[edgeID];
    bool covered = false;
    for (int solutionIndex = 0; solutionIndex < sizeOfKernelSolution + (depthOfLeaf+1)*2; ++solutionIndex){
        covered |= (edge.x == soln[solutionIndex]);
        covered |= (edge.y == soln[solutionIndex]);
    }
    /*
    // If you decide to use sm for dynamic verts
    for (int solutionIndex = 0; solutionIndex < counter; ++solutionIndex){
        covered |= (edge.x == soln[solutionIndex] || edge.y == soln[solutionIndex]);
    }
    */
    #ifndef NDEBUG
    if (threadIdx.x == 0){
        printf("LI %d Dyn soln\n", leafIndex);
        for (int index = 0; index < UBDyn; ++index)
            printf("%d ", ddynamicallyaddedvertices[index]);
        printf("\n");
    }
    __syncthreads(); // put warp results in shared mem
    #endif
    for (int index = 0; index < UBDyn; ++index){
        covered |= (edge.x == ddynamicallyaddedvertices[index]);
        covered |= (edge.y == ddynamicallyaddedvertices[index]);
    }
    // Maybe do a warp shuffle and only 1 atomic add per block
    if (!covered){
        #ifndef NDEBUG
        printf("Edge (%d - %d) uncovered\n", edge.x, edge.y);
        #endif
        atomicAdd(uncoverededges, 1);
    } 
    #ifndef NDEBUG
    else {
        printf("Edge (%d - %d) covered\n", edge.x, edge.y);
    }
    #endif
}

__global__ void PrintSolutionArray(int solutionSize,
                                int * dsolution){
    for (int i = 0; i < solutionSize; ++i){
        printf("%d ", dsolution[i]);
    }
}

// Single threaded; could accelerate eventually.
__global__ void FillSolutionArray(int leafIndex,
                                int * dsolution,
                                int sizeOfKernelSolution,
                                int2 * dsearchtree,
                                int * dnumberofdynamicallyaddedvertices,
                                int * ddynamicallyaddedvertices){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    int leafIndexSoln = leafIndex;
    int2 nodeEntry;
    int counter = 0;
    int UBDyn = dnumberofdynamicallyaddedvertices[0];
    if (threadID == 0){
        while(leafIndexSoln != 0){
            nodeEntry = dsearchtree[leafIndexSoln];
            dsolution[sizeOfKernelSolution + counter] = nodeEntry.x;
            dsolution[sizeOfKernelSolution + counter + 1] = nodeEntry.y;
            //printf("Tree verts %d %d\n", nodeEntry.x, nodeEntry.y);
            if(leafIndexSoln % 3 == 0){
                --leafIndexSoln;
                leafIndexSoln = leafIndexSoln / 3;
            } else {
                leafIndexSoln = leafIndexSoln / 3;
            }
            counter += 2;
        }
        for (int index = 0; index < UBDyn; ++index){
            dsolution[sizeOfKernelSolution + counter + index] = ddynamicallyaddedvertices[index];
            //printf("Dynamic vertex %d\n", dsolution[counter+index]);
        }
    }
}
// Each block is a leaf node
// First it loads it's solution into shared memory.
// ADVANCED - If kPrime*sizeof(int) > shared memory limit, check in portions
// Each thread takes an edge, iterate over all edges,
// check of vertex a and vertex b is missing from soln
// if so, indicate in final position of shared memory.
// sync threads
// terminate prematurely if final position flag is set.
// this way we can check leaf nodes in parallel, without needing 
// an edge status array.
/*
__global__ void EvaluateLeafNodesV2(int nrEdges,
                                    mtc::Edge * dedges, 
                                    int sizeOfSearchTree,
                                    int depthOfSearchTree,
                                    int2 * dsearchtree){
    extern __shared__ int soln[];
    const int leafIndex = blockIdx.x;
    int thisThreadsSearchTreeNode;
    int2 nodeEntry;

    int tid = threadIdx.x;
    unsigned mask = 0xFFFFFFFFU;
    int lane = threadIdx.x % warpSize;
    int warpID = threadIdx.x / warpSize;
    // Load solution into shared memory
    // Need depthOfSearch minus 1 to exclude root
    for (int numberOfLevelsToAscend = threadIdx.x; numberOfLevelsToAscend < depthOfSearchTree-1; numberOfLevelsToAscend += blockDim.x){
        thisThreadsSearchTreeNode = leafIndex / pow (3.0, numberOfLevelsToAscend);
        nodeEntry = dsearchtree[thisThreadsSearchTreeNode];
        soln[2*numberOfLevelsToAscend] = nodeEntry.x;
        soln[2*numberOfLevelsToAscend + 1] = nodeEntry.y;
    }
    int covered;
    for (int e = 0; e < nrEdges; e++){
        Edge & edge = dedges[e];
        for (int vertexInAnswer = threadIdx.x; 
                    vertexInAnswer < 2*(depthOfSearchTree-1); 
                    vertexInAnswer += blockDim.x){
            covered |= edge.x == soln[vertexInAnswer] || edge.y == soln[vertexInAnswer];
        }
        // 1st warp-shuffle reduction
        for (int offset = warpSize/2; offset > 0; offset >>= 1)
            covered |= __shfl_down_sync(mask, covered, offset);
        if (lane == 0) sdata[warpID] = covered;
        __syncthreads(); // put warp results in shared mem
        // hereafter, just warp 0
        if (warpID == 0){
            // reload val from shared mem if warp existed
            val = (tid < blockDim.x/warpSize)?sdata[lane]:0;
            // final warp-shuffle reduction
            for (int offset = warpSize/2; offset > 0; offset >>= 1)
            val += __shfl_down_sync(mask, val, offset);
            if (tid == 0) atomicAdd(dremainingedges, val);
        }
    }
}
*/
// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void DetectAndSetPendantPathsCase4(int nrVertices, 
                                                int kPrime,
                                                int *match, 
                                                int *dforwardlinkedlist, 
                                                int *dbackwardlinkedlist, 
                                                int * dedgestatus,
                                                int *dlength, 
                                                int *dnumberofdynamicallyaddedvertices,
                                                int *ddynamicallyaddedvertices){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    int dynamicIndex;

	// If not a head to a path of length 4, return (leaving the headindex == -1)
    if (threadID >= nrVertices || 
        dlength[threadID] != 1 || 
        dbackwardlinkedlist[threadID] != threadID) 
            return;

    int first = dforwardlinkedlist[threadID];
    int second = dforwardlinkedlist[first];

    // Color == 2 if blue vertex has no unmatched neighbors
    // This avoids iterating over all degrees, but it is possible
    // to miss some vertices which could be pendant but are red not blue.
    if (match[first] == 3){
        dynamicIndex = atomicAdd(&dnumberofdynamicallyaddedvertices[0], 1); 
        //ddynamicallyaddedvertices[dynamicIndex] = first;
        //SetEdges(first, dedgestatus);
    } else if (match[second] == 3){
        dynamicIndex = atomicAdd(&dnumberofdynamicallyaddedvertices[0], 1); 
        //ddynamicallyaddedvertices[dynamicIndex] = second;
        //SetEdges(second, dedgestatus);
    }

    if (match[first] == 3 || match[second] == 3){
        if (dynamicIndex < kPrime){
            if (match[first] == 3){
                ddynamicallyaddedvertices[dynamicIndex] = first;
            } else if (match[second] == 3){
                ddynamicallyaddedvertices[dynamicIndex] = second;
            }
            //printf("added pendant %d", dynamicIndex);
        } else {
            atomicSub(&dnumberofdynamicallyaddedvertices[0], 1);
        }
    }
}

// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void DetectAndSetPendantPathsCase3(int nrVertices, 
                                              int kPrime,
                                                int *match, 
                                                int *dforwardlinkedlist, 
                                                int *dbackwardlinkedlist, 
                                                int * dedgestatus,
                                                int *dlength, 
                                                int *dnumberofdynamicallyaddedvertices,
                                                int *ddynamicallyaddedvertices){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    int dynamicIndex;
    // If not a head to a path of length 4, return (leaving the headindex == -1)
    if (threadID >= nrVertices || 
        dlength[threadID] != 2 || 
        dbackwardlinkedlist[threadID] != threadID) 
            return;

    int first = dforwardlinkedlist[threadID];
    int second = dforwardlinkedlist[first];
    int third = dforwardlinkedlist[second];

    // Color == 2 if blue vertex has no unmatched neighbors
    // This avoids iterating over all degrees, but it is possible
    // to miss some vertices which could be pendant but are red not blue.
    if (match[first] == 3){
        dynamicIndex = atomicAdd(&dnumberofdynamicallyaddedvertices[0], 1); 
        //ddynamicallyaddedvertices[dynamicIndex] = first;
        //SetEdges(first, dedgestatus);
    } else if (match[third] == 3){
        dynamicIndex = atomicAdd(&dnumberofdynamicallyaddedvertices[0], 1); 
        //ddynamicallyaddedvertices[dynamicIndex] = third;
        //SetEdges(third, dedgestatus);
    }

    if (match[first] == 3 || match[third] == 3){
        if (dynamicIndex < kPrime){
            if (match[first] == 3){
                ddynamicallyaddedvertices[dynamicIndex] = first;
            } else if (match[third] == 3){
                ddynamicallyaddedvertices[dynamicIndex] = third;
            }
            //printf("added pendant %d", dynamicIndex);
        } else {
            atomicSub(&dnumberofdynamicallyaddedvertices[0], 1);
        }
    }
}

// Makes sense for BFS
// For DFS use Recursive Backtracking
__global__ void GetFrontierStatus(int nrNodes,
							int *active_frontier_status){

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
	const int numberOfLevelsToAscend = blockIdx.x/2;
    //if (threadIdx.x == 0){
    int thisBlocksSearchTreeNode = leafIndex / pow (3.0, numberOfLevelsToAscend);
    //}
    int2 verticesInNode = dsearchtree[thisBlocksSearchTreeNode];
    int i;
    if (blockIdx.x % 2 == 0)
        i = verticesInNode.x;
    else 
        i = verticesInNode.y;
    int2 indices = tex1Dfetch(neighbourRangesTexture, i);
    #ifndef NDEBUG
    if (threadIdx.x == 0){
        printf("thisBlocksSearchTreeNode %d\n", thisBlocksSearchTreeNode);
        printf("Setting vertex %d\n", i);
        printf("Turning off edges between %d and %d in col array\n",indices.x,indices.y);
    }
    #endif
    for (int j = indices.x + threadIdx.x; j < indices.y; j += blockDim.x){
        //const int ni = tex1Dfetch(neighboursTexture, j);
        //printf("Turning off edge %d which is index %d of the val array\n",ni,j);
        // Set out-edges
        dedgestatus[j] = 0;
    }   
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
    for (int j = indices_curr.x + threadIdx.x; j < indices_curr.y; j += blockDim.x){
        const int ni = tex1Dfetch(neighboursTexture, j);    
        const int2 indices_neighbor = tex1Dfetch(neighbourRangesTexture, ni);
          for (int j_n = indices_neighbor.x; j_n < indices_neighbor.y; ++j_n){
                const int nj = tex1Dfetch(neighboursTexture, j_n);       
                foundChild = i == nj;
                // Set in-edge
                // store edge status
                tmp = dedgestatus[j_n];
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
                // Doesnt work for some reason
                // dedgestatus[j_n] ^= (foundChild & tmp);

                if(foundChild && tmp)
                    dedgestatus[j] = 0;
        }
    } 
}


// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void BussKernelizationP1Kernel(int nrVertices, 
                                        int k, 
                                        int *ddegrees,
                                        int *dsolution,
                                        int *dsizeofkernelsolution){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= nrVertices) return;
    int degree = ddegrees[threadID];
    if (degree <= k) return;
    int solutionIndex = atomicAdd(&dsizeofkernelsolution[0], 1);
    // dsolution = new int[k];
    // Prevent oob
    if (solutionIndex >= k){
        return;
    }
    dsolution[solutionIndex] = threadID;
}

__global__ void BussKernelizationP2Kernel(int sizeOfKernelSolution,
                                        int *ddegrees,
                                        int *dremainingedges,
                                        int *dsolution){
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= sizeOfKernelSolution) return;
    int solnVertex = dsolution[threadID];
    int degree = ddegrees[solnVertex];
    int remainingedges = atomicSub(&dremainingedges[0], degree);
    //printf("Removed %d's %d edges : edges remaining %d\n", solnVertex, degree/2, remainingedges/2);
}

__device__ void SetEdges(   int vertexToInclude,
                            int * dedgestatus){

    int2 indices = tex1Dfetch(neighbourRangesTexture, vertexToInclude);
    for (int j = indices.x; j < indices.y; j += 1){
        //const int ni = tex1Dfetch(neighboursTexture, j);
        //printf("Turning off edge %d which is index %d of the val array\n",ni,j);
        // Set out-edges
        dedgestatus[j] = 0;
    }   
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
    const int2 indices_curr = tex1Dfetch(neighbourRangesTexture, vertexToInclude);
    for (int j = indices_curr.x; j < indices_curr.y; j += 1){
        const int ni = tex1Dfetch(neighboursTexture, j);    
        const int2 indices_neighbor = tex1Dfetch(neighbourRangesTexture, ni);
          for (int j_n = indices_neighbor.x; j_n < indices_neighbor.y; ++j_n){
                const int nj = tex1Dfetch(neighboursTexture, j_n);       
                foundChild = vertexToInclude == nj;
                // Set in-edge
                // store edge status
                tmp = dedgestatus[j_n];
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
                // Doesnt work for some reason
                // dedgestatus[j_n] ^= (foundChild & tmp);

                if(foundChild && tmp)
                    dedgestatus[j] = 0;
        }
    } 
}


__global__ void CalculateDegrees(
                        int nrVertices,
                        int * dedgestatus,
                        int * ddegrees){

	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
    if (threadID >= nrVertices ) return;
    int sum = 0;
    int2 indices = tex1Dfetch(neighbourRangesTexture, threadID);
    for (int j = indices.x; j < indices.y; ++j){
        sum += dedgestatus[j];
    }
    ddegrees[threadID] = sum;
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

// Template this to do any type of tree
// binary, ternary, quaternary, ...
int4 CalculateLeafOffsets(              int leafIndex,
                                        int fullpathcount){

    unsigned int leavesToProcess = fullpathcount;
    unsigned int leavesFromIncompleteLvl = 1;
    unsigned int leavesFromCompleteLvl = 1;

    if (leavesToProcess == 0)
        return make_int4( leafIndex,
                          leafIndex,
                          leafIndex,
                          leafIndex);

    unsigned int n_com = floor(logf(2*leavesToProcess + 1) / logf(3));
    unsigned int n_inc = ceil(logf(2*leavesToProcess + 1) / logf(3));

    for (unsigned int i = 1; i <= n_inc; ++i)
        leavesFromIncompleteLvl*=3;

    for (unsigned int i = 1; i <= n_com; ++i)
        leavesFromCompleteLvl*=3;
    //float nf = logf(2*leavesToProcess + 1) / logf(3);
    unsigned int arbitraryParameter = 3*((3*leafIndex)+1);
    // At high powers, the error of transendental powf causes bugs.
    //unsigned int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

    // Discrete calculation without trancendentals
    unsigned int leftMostLeafIndexOfIncompleteLevel = (2*arbitraryParameter+3);
    unsigned int multiplicandn_inc = 1;
    for (unsigned int i = 1; i < n_inc; ++i)
        multiplicandn_inc*=3;
    leftMostLeafIndexOfIncompleteLevel*=multiplicandn_inc;
    leftMostLeafIndexOfIncompleteLevel-=3;
    leftMostLeafIndexOfIncompleteLevel/=6;

    unsigned int leftMostLeafIndexOfCompleteLevel = (2*arbitraryParameter+3);
    unsigned int multiplicandn_com = 1;
    for (unsigned int i = 1; i < n_com; ++i)
        multiplicandn_com*=3;
    leftMostLeafIndexOfCompleteLevel*=multiplicandn_com;
    leftMostLeafIndexOfCompleteLevel-=3;
    leftMostLeafIndexOfCompleteLevel/=6;

    unsigned int treeSizeNotIncludingThisLevel = (1.0 - multiplicandn_inc)/(1.0 - 3.0); 
    // At high powers, the error of transendental powf causes bugs. 
    //unsigned int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, ((n+1)-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons???
    unsigned int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    //unsigned int internalLeafIndex = leavesToProcess - treeSizeNotIncludingThisLevel;
    unsigned int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

    #ifndef NDEBUG
    printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, n_com);
    printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, n_inc);
    printf("Leaves %d, leavesFromCompleteLvl %d\n",leavesToProcess, leavesFromCompleteLvl);
    printf("Leaves %d, leavesFromIncompleteLvl %d\n",leavesToProcess, leavesFromIncompleteLvl);
    printf("Leaves %d, leftMostLeafIndexOfIncompleteLevel %d\n",leavesToProcess, leftMostLeafIndexOfIncompleteLevel);
    #endif
    // Grow tree leftmost first, so put the incomplete level first.
    // Shape of leaves
    //CL    -     -    o o o 
    //IL  o o o o o o
    unsigned int clb;
    unsigned int cub;

    if (n_com == n_inc){
        clb = levelOffset + 3;
        cub = levelOffset + 3;
    } else {
        clb = (levelOffset + 2)/3;
        cub = leftMostLeafIndexOfCompleteLevel + leavesFromCompleteLvl;
    }
    // Grow tree leftmost first, so put the incomplete level first.
    // Shape of leaves
    //CL    -     -    o o o 
    //IL  o o o o o o
    return make_int4(leftMostLeafIndexOfIncompleteLevel,
                     levelOffset + 3,
                     clb,
                     cub);
}

void VCGPU::CopyMatchingBackToHost(std::vector<int> & match){
	//Copy obtained matching on the device back to the host.
	if (cudaMemcpy(&match[0], dmatch, sizeof(int)*graph.nrVertices, cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		cerr << "Unable to retrieve data!" << endl;
		throw exception();
	}
}