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
 * @brief Defines the Policy for representing the bounded search tree.
 * @tparam T Datatype of indices in the search tree nodes.  Depends on number of vertices in graph.
 */

/**
 * @brief The tree is explicitly allocated. O(3^k) memory requirement.
 * @tparam T Datatype of indices in the search tree nodes.  Depends on number of vertices in graph.
 */
template <class T>
struct ExplicitSearchTree
{
static T* Create()
{
return new T;
}

    std::vector<int2> searchtree;

};

/**
 * @brief The tree is generated by saving the paths and logical indexing. O(k) memory requirement.
 * @tparam T Datatype of indices in the search tree nodes.  Depends on number of vertices in graph.
 */
template <class T>
struct ImplicitSearchTree
{
static T* Create()
{
return new T;
}
    
    T * rows;
    T * columns;

};

#endif