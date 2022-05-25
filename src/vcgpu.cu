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

void VCGPU::findCover(int nrVertices, int threadsPerBlock, int *dforwardlinkedlist, int *dbackwardlinkedlist, int *dmatch, int *dlength, int *dheadlist, int *dheadbool)
{
	int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;

	SetHeadBool<<<blocksPerGrid, threadsPerBlock>>>(nrVertices, dbackwardlinkedlist, dheadbool);
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


__global__ void SetHeadBool(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dheadbool){
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	if (threadID >= nrVertices) return;

    dheadbool[threadID] = dbackwardlinkedlist[threadID] == threadID;
}