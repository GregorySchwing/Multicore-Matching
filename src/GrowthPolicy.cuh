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
#ifndef GROWTH_POLICY_H
#define GROWTH_POLICY_H

#include "graph.h"
#include "BussKernelization.cuh"
#include "TreeBuilder.cuh"
#include "TritArrayMaker.h"


template <class T>
struct Serial
{

Serial(const mtc::Graph &_graph, 
              const int &_threadsPerBlock, 
              const unsigned int &_barrier,
              int _k): 
              matcher(_graph, _threadsPerBlock, _barrier),
              graph(_graph),
              threadsPerBlock(_threadsPerBlock),
              k(_k),
              kPrime(_k){
    //cudaMemcpy(dDegrees, graph.degrees.data(), sizeof(int)*graph.nrVertices, cudaMemcpyHostToDevice);
    //bk = new BussKernelization(_graph, _threadsPerBlock, _barrier, _k, solutionCantExist);
    hostTreeRows = new T[k+1];
    hostTreeColumns = new T[4*k];
    hostDynamicRows = new T[k+1];
    hostDynamicColumns = new T[k];
    hostKernelRows = new T[k+1];
    hostKernelColumns = new T[k];
    if (cudaMalloc(&deviceTreeRows, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMalloc(&deviceTreeColumns, sizeof(T)*(4*k)) != cudaSuccess || 
        cudaMalloc(&deviceDynamicRows, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMalloc(&deviceDynamicColumns, sizeof(T)*(k)) != cudaSuccess || 
        cudaMalloc(&deviceKernelRows, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMalloc(&deviceKernelColumns, sizeof(T)*(k)) != cudaSuccess||
        cudaMalloc(&deviceRemainingEdges, sizeof(T)) != cudaSuccess)
    {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}

    if (cudaMemset(deviceTreeRows, 0, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMemset(deviceTreeColumns, 0, sizeof(T)*(4*k)) != cudaSuccess || 
        cudaMemset(deviceDynamicRows, 0, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMemset(deviceDynamicColumns, 0, sizeof(T)*(k)) != cudaSuccess || 
        cudaMemset(deviceKernelRows, 0, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMemset(deviceKernelColumns, 0, sizeof(T)*(k)) != cudaSuccess ||
        cudaMemset(deviceRemainingEdges, 0, sizeof(T)) != cudaSuccess){
        std::cerr << "Error clearing memory!" << std::endl;
		throw std::exception();
    }
    printf("Calling Serial's BussKernelization::PerformBussKernelization\n");

    BussKernelization::PerformBussKernelization(graph.nrVertices, threadsPerBlock, k, kPrime, 1, matcher.ddegrees, deviceKernelRows, deviceKernelColumns, deviceRemainingEdges, solutionCantExist);
    printf("Returned from Serial's BussKernelization::PerformBussKernelization\n");
}

void FindCover(cpp_int root, int recursiveStackDepth, bool & foundSolution){
    printf("Called Serial FC\n");
    if (foundSolution){
            printf("Found SOLN\n");

        return;
    }


    // Lazy way to do this.
    cudaMemcpy(&numberofkernelvertices, &deviceKernelRows[recursiveStackDepth], sizeof(T)*1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&numberoftreevertices, &deviceTreeRows[recursiveStackDepth], sizeof(T)*1, cudaMemcpyDeviceToHost);
    cudaMemcpy(&numberofdynamicallyaddedvertices, &deviceDynamicRows[recursiveStackDepth], sizeof(T)*1, cudaMemcpyDeviceToHost);
    std::cout << "calling FC in leaf " << root << " recursiveStackDepth " << recursiveStackDepth << std::endl
    << "numberofkernelvertices " << numberofkernelvertices << " numberoftreevertices " << numberoftreevertices << std::endl
    << "numberofdynamicallyaddedvertices " << numberofdynamicallyaddedvertices << " numberoftreevertices " << numberoftreevertices << std::endl;



    if (numberofkernelvertices+numberoftreevertices+numberofdynamicallyaddedvertices <= k) {
        Match(root);
        cudaMemcpy(&deviceTreeRows[recursiveStackDepth], &deviceTreeRows[recursiveStackDepth-1], sizeof(T)*1, cudaMemcpyDeviceToDevice);
        TreeBuilder::PopulateTree(graph.nrVertices,
                                    threadsPerBlock,
                                    k,
                                    &deviceTreeRows[recursiveStackDepth], 
                                    deviceTreeColumns, 
                                    matcher.dforwardlinkedlist, 
                                    matcher.dbackwardlinkedlist, 
                                    matcher.dlength);
        T temp[2];
        cudaMemcpy(temp, &deviceTreeRows[recursiveStackDepth-1], sizeof(T)*2, cudaMemcpyDeviceToHost);
        int newLeaves = temp[1] - temp[0];
        printf("new leaves %d\n", newLeaves);
        cpp_int numNewLeaves = TritArrayMaker::large_pow(newLeaves);
        std::cout << "numNewLeaves " << numNewLeaves << std::endl;
        
        for (cpp_int leaf = 0; leaf < numNewLeaves; ++leaf){
            std::cout << "calling FC in leaf " << leaf << " recursiveStackDepth " << recursiveStackDepth << std::endl;
            FindCover(leaf, ++recursiveStackDepth, foundSolution);
            std::cout << "Returned from FC in leaf " << leaf << " recursiveStackDepth " << recursiveStackDepth << std::endl;
        }
    } else {
    }
}



template <typename U>
static void PopulateTree(U* param1, int param2);

    private:
        void Match(cpp_int leafIndex);
        mtc::GraphMatchingGeneralGPURandom matcher;
		const mtc::Graph &graph;
        const int &threadsPerBlock;

        int numberofkernelvertices = 0;
        int numberofdynamicallyaddedvertices = 0;
        int numberoftreevertices = 0;
        const int k;
        int kPrime;
        T * hostTreeRows;
        T * hostTreeColumns;
        T * deviceTreeRows;
        T * deviceTreeColumns;

        T * hostDynamicRows;
        T * hostDynamicColumns;
        T * deviceDynamicRows;
        T * deviceDynamicColumns;

        T * hostKernelRows;
        T * hostKernelColumns;
        T * deviceKernelRows;
        T * deviceKernelColumns;

        T * hostRemainingEdges;
        T * deviceRemainingEdges;

        bool solutionCantExist;

};

template <class T>
struct Parallel
{
// Going to use view to automatically stride each stream.
// https://stackoverflow.com/questions/4754763/object-array-initialization-without-default-constructor
Parallel(const mtc::Graph &_graph, 
              const int &_threadsPerBlock, 
              const unsigned int &_barrier,
              int _k):
              graph(_graph),
              threadsPerBlock(_threadsPerBlock),
              k(_k),
              kPrime(_k){
    raw_memory = operator new[](NUM_STREAMS * sizeof(mtc::GraphMatchingGeneralGPURandom));
    mtc::GraphMatchingGeneralGPURandom *matcher = static_cast<mtc::GraphMatchingGeneralGPURandom *>(raw_memory);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        new(&matcher[i]) mtc::GraphMatchingGeneralGPURandom(_graph, _threadsPerBlock, _barrier);
    }
    hostTreeRows = new T[NUM_STREAMS*(k+1)];
    hostTreeColumns = new T[NUM_STREAMS*(4*k)];
    hostDynamicRows = new T[NUM_STREAMS*(k+1)];
    hostDynamicColumns = new T[NUM_STREAMS*k];
    hostRemainingEdges = new T[NUM_STREAMS];

    if (cudaMalloc(&deviceTreeRows, sizeof(T)*(NUM_STREAMS*(k+1))) != cudaSuccess || 
        cudaMalloc(&deviceTreeColumns, sizeof(T)*(NUM_STREAMS*(4*k))) != cudaSuccess || 
        cudaMalloc(&deviceDynamicRows, sizeof(T)*(NUM_STREAMS*(k+1))) != cudaSuccess || 
        cudaMalloc(&deviceDynamicColumns, sizeof(T)*(NUM_STREAMS*k)) != cudaSuccess ||
        cudaMalloc(&deviceRemainingEdges, sizeof(T)*(NUM_STREAMS)) != cudaSuccess)
    {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}

    //for (int i = 0; i < NUM_STREAMS; ++i) {
        printf("Calling Parallel's BussKernelization::PerformBussKernelization\n");
        BussKernelization::PerformBussKernelization(graph.nrVertices, threadsPerBlock, k, kPrime, 1, matcher->ddegrees, deviceKernelRows, deviceKernelColumns, deviceRemainingEdges, solutionCantExist);
        printf("Returned from Parallel's BussKernelization::PerformBussKernelization\n");

    //}
}
    private:
		const mtc::Graph &graph;
        const int &threadsPerBlock;
        /// @brief Determined by the size of the graph, and the number of replicates that can be fit in device memory.
        int NUM_STREAMS = 5;
        void *raw_memory;
        /// @brief Array of matchers, one for each stream.
        mtc::GraphMatchingGeneralGPURandom * matcher;
        // Make these arrays
        int numberofkernelvertices = 0;
        int numberofdynamicallyaddedvertices = 0;
        int numberoftreevertices = 0;
        const int k;
        int kPrime;
        T * hostTreeRows;
        T * hostTreeColumns;
        T * deviceTreeRows;
        T * deviceTreeColumns;

        T * hostDynamicRows;
        T * hostDynamicColumns;
        T * deviceDynamicRows;
        T * deviceDynamicColumns;

        T * hostKernelRows;
        T * hostKernelColumns;
        T * deviceKernelRows;
        T * deviceKernelColumns;

        T * hostRemainingEdges;
        T * deviceRemainingEdges;

        bool solutionCantExist;


void FindCover(){
    printf("Called Parallel FindCover\n");
}

static T* Create()
{
return new T;
}



template <typename U>
static void PopulateTree(U* param1, int param2);
};

template <class T>
template <typename U>
void Serial<T>::PopulateTree(U* param1, int param2){
    kernel_wrapper(param1, param2);
}

template <class T>
void Serial<T>::Match(cpp_int leafIndex){
    //Initialise timers.
    cudaEvent_t t0, t1, t2, t3;
    float time0, time1;

    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventCreate(&t2);
    cudaEventCreate(&t3);
    
    cudaEventRecord(t0, 0);
    cudaEventSynchronize(t0);
    printf("Called Match Serial\n");
    matcher.performMatching(matcher.dmatch, t1, t2, numberofkernelvertices, deviceKernelColumns, numberoftreevertices, deviceTreeColumns, numberofdynamicallyaddedvertices, deviceDynamicColumns, leafIndex);
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
#endif