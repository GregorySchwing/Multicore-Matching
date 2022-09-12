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
#ifndef TREE_REPRESENTATION_POLICY_H
#define TREE_REPRESENTATION_POLICY_H

/**
 * @brief The tree is explicitly allocated. O(3^k) memory requirement.
 * @tparam T Datatype of indices in the search tree nodes.  Depends on number of vertices in graph.
 */
template <class T>
struct ExplicitTree
{

void Create(int k)
{
    long long sizeOfTree = CalculateSpaceForDesiredNumberOfLevels(k);
    searchtree.resize(sizeOfTree);
    if (cudaMalloc(&dsearchtree, sizeof(T)*(2*sizeOfTree)) != cudaSuccess)
    {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}
}

    std::vector<T> searchtree;
    T * dsearchtree;
    long long CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels);
};

template <class T>
long long ExplicitTree<T>::CalculateSpaceForDesiredNumberOfLevels(int NumberOfLevels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i <= NumberOfLevels; ++i){
        summand += pow (3.0, i);
    }
    return summand;
}

/**
 * @brief The tree is generated by saving the paths in columns of a CSR, with each recursive match call \
 * in a row. Then generating the leaves by logical indexing. O(k) memory requirement.
 * @tparam T Datatype of indices in the search tree nodes.  Depends on number of vertices in graph.
 */
template <class T>
struct SerialImplicitTree
{
  
void Create(int k)
{
    hostTreeRows = new T[k+1];
    hostTreeColumns = new T[4*k];
    hostDynamicRows = new T[k+1];
    hostDynamicColumns = new T[k];
    if (cudaMalloc(&deviceTreeRows, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMalloc(&deviceTreeColumns, sizeof(T)*(4*k)) != cudaSuccess || 
        cudaMalloc(&deviceDynamicRows, sizeof(T)*(k+1)) != cudaSuccess || 
        cudaMalloc(&deviceDynamicColumns, sizeof(T)*(k)) != cudaSuccess)
    {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}
    
}
 
    T * hostTreeRows;
    T * hostTreeColumns;
    T * deviceTreeRows;
    T * deviceTreeColumns;

    T * hostDynamicRows;
    T * hostDynamicColumns;
    T * deviceDynamicRows;
    T * deviceDynamicColumns;

};

/**
 * @brief The tree is generated by saving the paths in columns of a CSR, with each recursive match call \
 * in a row. Then generating the leaves by logical indexing. O(k) memory requirement.
 * @tparam T Datatype of indices in the search tree nodes.  Depends on number of vertices in graph.
 */
template <class T>
struct ParallelImplicitTree
{
  
void Create(int k)
{
    hostTreeRows = new T[NUM_STREAMS*(k+1)];
    hostTreeColumns = new T[NUM_STREAMS*(4*k)];
    hostDynamicRows = new T[NUM_STREAMS*(k+1)];
    hostDynamicColumns = new T[NUM_STREAMS*k];
    if (cudaMalloc(&deviceTreeRows, sizeof(T)*(NUM_STREAMS*(k+1))) != cudaSuccess || 
        cudaMalloc(&deviceTreeColumns, sizeof(T)*(NUM_STREAMS*(4*k))) != cudaSuccess || 
        cudaMalloc(&deviceDynamicRows, sizeof(T)*(NUM_STREAMS*(k+1))) != cudaSuccess || 
        cudaMalloc(&deviceDynamicColumns, sizeof(T)*(NUM_STREAMS*k)) != cudaSuccess)
    {
		std::cerr << "Not enough memory on device!" << std::endl;
		throw std::exception();
	}
    
}
    /// @brief Determined by the size of the graph, and the number of replicates that can be fit in device memory.
    int NUM_STREAMS = 5;

    T * hostTreeRows;
    T * hostTreeColumns;
    T * deviceTreeRows;
    T * deviceTreeColumns;

    T * hostDynamicRows;
    T * hostDynamicColumns;
    T * deviceDynamicRows;
    T * deviceDynamicColumns;

};

#endif