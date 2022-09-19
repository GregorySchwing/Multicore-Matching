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
#ifndef TREE_BUILDER_H
#define TREE_BUILDER_H

class TreeBuilder {
    public:
        static void PopulateTree(int nrVertices, 
                                    int threadsPerBlock,
                                    int k, 
                                    int *deviceTreeRows, 
                                    int *deviceTreeColumns,
                                    int *deviceDynamicRows, 
                                    int *deviceDynamicColumns,
                                    int *dmatch, 
                                    int *dforwardlinkedlist, 
                                    int *dbackwardlinkedlist, 
                                    int *dlength);

};

#endif