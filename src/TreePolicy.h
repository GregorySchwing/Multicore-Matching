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
}

    std::vector<T> searchtree;
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
struct ImplicitTree
{
  
void Create(int k)
{
    rows = new T[k+1];
    columns = new T[4*k];
}
    
    T * rows;
    T * columns;

};

#endif