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

void VCGPU::findCover(int *dforwardlinkedlist, int *dbackwardlinkedlist, int *dmatch, int *dlength, int *dheadlist, int *dheadbool)
{

}

void VCGPU::SortByHeadBool(int *dheadlist, int *dheadbool)
{

}


__global__ void SetHeadBool(int nrVertices,
							int *dbackwardlinkedlist,
                            int* dheadlist,
                            int* dheadbool){
	//Determine blue and red groups using MD5 hashing.
	//Based on the Wikipedia MD5 hashing pseudocode (http://en.wikipedia.org/wiki/MD5).
	const int threadID = blockIdx.x*blockDim.x + threadIdx.x;
	if (i >= nrVertices) return;

    dheadbool[threadID] = dbackwardlinkedlist[threadID] == threadID;
}