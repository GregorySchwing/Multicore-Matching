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

__global__ void SetHeadIndex(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dheadbool);

__global__ void ReducePathLengths(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dlength,
                            int* dreducedlength);

namespace mtc
{
class VCGPU
{
	public:
		VCGPU(const Graph &, const int &_threadsPerBlock);
		~VCGPU();
		
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

        // VC arrays
        int *dedgestatus, *ddegrees;

	protected:
		const Graph &graph;
        const int threadsPerBlock;
		int2 *dneighbourRanges;
		int *dneighbours;
};

};

#endif
