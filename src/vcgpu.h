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

namespace mtc
{
__global__ void SetHeadBool(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dheadlist,
                            int* dheadbool);
class VCGPU
{
	public:
		VCGPU();
		~VCGPU();
		

		void findCover(int nrVertices, int threadsPerBlock, int *dforwardlinkedlist, int *dbackwardlinkedlist, int *dmatch, int *dlength, int *dheadlist, int *dheadbool);
		void SortByHeadBool(int *dheadlist, int *dheadbool);

};

#endif
