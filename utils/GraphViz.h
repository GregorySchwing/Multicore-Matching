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

#include "../DotWriter/lib/DotWriter.h"
//#include "../DotWriter/lib/Enums.h"
#include <set>
#include <sstream>

class GraphViz {
    public:
        GraphViz();
        void DrawInputGraphColored();
        void DrawSearchTree();

    private:
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

        void writeGraphViz(std::vector<int> & match, 
					const Graph & g,
					const string &fileName_arg,  
					std::vector<int> & fll,
					std::vector<int> & bll);
};
#endif