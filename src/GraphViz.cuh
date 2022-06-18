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
#ifndef GRAPHVIZ_H
#define GRAPHVIZ_H
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include "../DotWriter/lib/DotWriter.h"
#include <set>
#include <sstream>
#include <string>

#include "graph.h"

class GraphViz {
    public:
        GraphViz();
        void DrawInputGraphColored(const mtc::Graph &_graph, 
									thrust::device_vector<int> & dmatch,
									thrust::device_vector<int> & dfll,
									thrust::device_vector<int> & dbll,
									int iter);
        void DrawSearchTree();

    private:
        thrust::host_vector<int> match;
        thrust::host_vector<int> fll;
        thrust::host_vector<int> bll;

        DotWriter::RootGraph * inputGraph;
        DotWriter::RootGraph * searchTree;

		std::string subgraph1 = "linearforest";
	    std::string subgraph2 = "fullgraph";

        DotWriter::Subgraph * linearforestgraph;
        DotWriter::Subgraph * fullgraph;

        std::map<std::string, DotWriter::Node *> linearForestNodeMap;    
        std::map<std::string, DotWriter::Node *> fullGraphNodeMap;    

        int curr, next;
        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1;
        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2;

        void writeGraphViz(thrust::host_vector<int> & match, 
					const mtc::Graph & g,
					const std::string &fileName_arg,  
					thrust::host_vector<int> & fll,
					thrust::host_vector<int> & bll);
};
#endif