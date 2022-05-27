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
//#include "cub/cub.cuh"

using namespace std;
using namespace mtc;

//==== Kernel variables ====
__device__ int dkeepMatching;

texture<int2, cudaTextureType1D, cudaReadModeElementType> neighbourRangesTexture;
texture<int, cudaTextureType1D, cudaReadModeElementType> neighboursTexture;
texture<float, cudaTextureType1D, cudaReadModeElementType> weightsTexture;

VCGPU::VCGPU(const Graph &_graph, const int &_threadsPerBlock, const unsigned int &_barrier) :
		graph(_graph),
        threadsPerBlock(_threadsPerBlock),
        barrier(_barrier),
		matcher(_graph, _threadsPerBlock, _barrier)
{
    if (cudaMalloc(&dedgestatus, sizeof(int)*graph.nrEdges) != cudaSuccess || 
        cudaMalloc(&ddegrees, sizeof(int)*graph.nrVertices) != cudaSuccess)
	{
		cerr << "Not enough memory on device!" << endl;
		throw exception();
	}
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(dedgestatus),  1, size_t(graph.nrEdges));
    cuMemsetD32(reinterpret_cast<CUdeviceptr>(ddegrees),  0, size_t(graph.nrVertices));
	//Setup textures.
    /*
	cudaChannelFormatDesc neighbourRangesTextureDesc = cudaCreateChannelDesc<int2>();

	neighbourRangesTexture.addressMode[0] = cudaAddressModeWrap;
	neighbourRangesTexture.filterMode = cudaFilterModePoint;
	neighbourRangesTexture.normalized = false;
	cudaBindTexture(0, neighbourRangesTexture, (void *)dneighbourRanges, neighbourRangesTextureDesc, sizeof(int2)*graph.neighbourRanges.size());
	
	cudaChannelFormatDesc neighboursTextureDesc = cudaCreateChannelDesc<int>();

	neighboursTexture.addressMode[0] = cudaAddressModeWrap;
	neighboursTexture.filterMode = cudaFilterModePoint;
	neighboursTexture.normalized = false;
	cudaBindTexture(0, neighboursTexture, (void *)dneighbours, neighboursTextureDesc, sizeof(int)*graph.neighbours.size());

	//Perform matching.
	int blocksPerGrid = (graph.nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    InitDegrees<<<blocksPerGrid, threadsPerBlock>>>(graph.nrVertices, ddegrees);
    */

}

VCGPU::~VCGPU(){
    cudaFree(dedgestatus);
    cudaFree(ddegrees);
	cudaUnbindTexture(neighboursTexture);
	cudaUnbindTexture(neighbourRangesTexture);
}

void VCGPU::GetLengthStatistics(int nrVertices, int threadsPerBlock, int *dbackwardlinkedlist, int *dlength, int *dreducedlength)
{
	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
	ReducePathLengths<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, dbackwardlinkedlist, dlength, dreducedlength);
}

void VCGPU::findCover(int nrVertices, int threadsPerBlock, int *dforwardlinkedlist, int *dbackwardlinkedlist, int *dmatch, int *dlength, int *headindex)
{
	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;

	SetHeadIndex<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, dbackwardlinkedlist, headindex);
}

void VCGPU::SortByHeadBool(int nrVertices,
                                int * dheadbool,
                                int * dheadboolSorted,
                                int * dheadlist,
                                int * dheadlistSorted){
    // Declare, allocate, and initialize device-accessible pointers for sorting data
    // numberOfAtoms            e.g., 7
    // mapParticleToCell        e.g., [8, 6, 7, 5, 3, 0, 9]
    // mapParticleToCellSorted  e.g., [        ...        ]
    // particleIndices          e.g., [0, 1, 2, 3, 4, 5, 6]
    // particleIndicesSorted    e.g., [        ...        ]
    // Determine temporary device storage requirements
    /*
    int num_items = nrVertices;
    int  *d_keys_in = dheadbool;
    int  *d_keys_out = dheadboolSorted;       
    int  *d_values_in = dheadlist;    
    int  *d_values_out = dheadlistSorted;   
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // Allocate temporary storage
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    // Run sorting operation
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out, d_values_in, d_values_out, num_items);
    // mapParticleToCellSorted        <-- [0, 3, 5, 6, 7, 8, 9]
    // particleIndicesSorted          <-- [5, 4, 3, 1, 2, 0, 6]
*/
}


__global__ void SetHeadIndex(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dheadbool){
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	if (threadID >= nrVertices) return;

    dheadbool[threadID] -= dbackwardlinkedlist[threadID] != threadID;
}

__global__ void ReducePathLengths(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dlength,
                            int* dreducedlength){}

__global__ void SetEdges(const int nrVertices,
                        int * dedgestatus,
                        int * ddegrees){

	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= nrVertices) return;

    const int2 indices = tex1Dfetch(neighbourRangesTexture, i);
    for (int j = indices.x; j < indices.y; ++j){
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
    for (int j = indices_curr.x; j < indices_curr.y; ++j){
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