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

#include "TreeBuilder.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <iostream>

// Alternative to sorting the full paths.  The full paths are indicated by a value >= 0.
__global__ void PopulateTreeKernel(int nrVertices, 
                                int k, 
                                int *deviceTreeRows, 
                                int *deviceTreeColumns, 
                                int *dforwardlinkedlist, 
                                int *dbackwardlinkedlist, 
                                int *dlength){
const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	// If not a head to a path of length 4, return (leaving the headindex == -1)
    if (threadID >= nrVertices || 
        dlength[threadID] != 3 || 
        dbackwardlinkedlist[threadID] != threadID) 
            return;

    int pathIndex = atomicAdd(&deviceTreeRows[0], 1);

    if (pathIndex >= k)
        return;

    int first = threadID;
    int second = dforwardlinkedlist[first];
    int third = dforwardlinkedlist[second];
    int fourth = dforwardlinkedlist[third];

    //printf("threadID %d wrote %d %d %d %d\n", threadID, first, second, third, fourth);

    // Test from root for now, this code can have an arbitrary root though
    deviceTreeColumns[4*pathIndex + 0] = first;
    deviceTreeColumns[4*pathIndex + 1] = second;
    deviceTreeColumns[4*pathIndex + 2] = third; 
    deviceTreeColumns[4*pathIndex + 3] = fourth;       
}
void TreeBuilder::PopulateTree(int nrVertices, 
                                int threadsPerBlock, 
                                int k, 
                                int *deviceTreeRows, 
                                int *deviceTreeColumns, 
                                int *dforwardlinkedlist, 
                                int *dbackwardlinkedlist, 
                                int *dlength){
    int blocksPerGrid = (nrVertices + threadsPerBlock - 1)/threadsPerBlock;
    PopulateTreeKernel<<<blocksPerGrid, threadsPerBlock>>>(nrVertices,
                                                            k, 
                                                            deviceTreeRows, 
                                                            deviceTreeColumns,
                                                            dforwardlinkedlist,
                                                            dbackwardlinkedlist, 
                                                            dlength);
}