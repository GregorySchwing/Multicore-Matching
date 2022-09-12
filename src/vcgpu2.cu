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
#include "vcgpu2.cuh"

#define nTPB 256

template <typename T>
__global__ void myKernel(T* param1, int param2){

  int i = threadIdx.x+blockDim.x*blockIdx.x;
  if (i < param2){
    param1[i] += (T)param2;
  }
}

template <typename T>
void kernel_wrapper(T* param1, int param2){
  myKernel<<<(param2+nTPB-1)/nTPB,nTPB>>>(param1, param2);
  cudaDeviceSynchronize();
}