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
              k(_k){
    sizeOfKernelSolution = numoftreeverts = numberofdynamicallyaddedvertices = 0;

}

void FindCover(int root, int recursiveStackDepth, bool & foundSolution){
    printf("Called Serial FC\n");
    if (foundSolution){
            printf("Found SOLN\n");

        return;
    }
    if (sizeOfKernelSolution+numoftreeverts+numberofdynamicallyaddedvertices <= k) {
                        printf("<= k\n");

        Match(root);
    } else {
                printf("> k\n");
    }

}


template <typename U>
static void PopulateTree(U* param1, int param2);

    private:
        void Match(int leafIndex);
        mtc::GraphMatchingGeneralGPURandom matcher;
		const mtc::Graph &graph;
        const int &threadsPerBlock;
        int sizeOfKernelSolution, numoftreeverts, numberofdynamicallyaddedvertices, k;
};

template <class T>
struct Parallel
{

// https://stackoverflow.com/questions/4754763/object-array-initialization-without-default-constructor
Parallel(const mtc::Graph &_graph, 
              const int &_threadsPerBlock, 
              const unsigned int &_barrier,
              int _k):
              graph(_graph),
              threadsPerBlock(_threadsPerBlock),
              k(_k){
    raw_memory = operator new[](NUM_STREAMS * sizeof(mtc::GraphMatchingGeneralGPURandom));
    mtc::GraphMatchingGeneralGPURandom *matcher = static_cast<mtc::GraphMatchingGeneralGPURandom *>(raw_memory);
    for (int i = 0; i < NUM_STREAMS; ++i) {
        new(&matcher[i]) mtc::GraphMatchingGeneralGPURandom(_graph, _threadsPerBlock, _barrier);
    }

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
        int sizeOfKernelSolution, numoftreeverts, numberofdynamicallyaddedvertices;
        int k;


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
void Serial<T>::Match(int leafIndex){
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
    //matcher.performMatching(dmatch, t1, t2, dsearchtree, ddynamicallyaddedvertices, dnumberofdynamicallyaddedvertices, sizeOfKernelSolution, dsolution, leafIndex);
    
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