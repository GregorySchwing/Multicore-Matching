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

#include "GraphViz.cuh"

#define SSTR( x ) static_cast< std::ostringstream & >( \
        ( std::ostringstream() << std::dec << x ) ).str()

GraphViz::GraphViz(){
	inputGraph = new DotWriter::RootGraph(false, "graph");
    linearforestgraph = inputGraph->AddSubgraph(subgraph1);
    fullgraph = inputGraph->AddSubgraph(subgraph2);

	searchTreeGraph = new DotWriter::RootGraph(false, "graph");
    searchtreesubgraph = searchTreeGraph->AddSubgraph(subgraph3);
}

void GraphViz::DrawInputGraphColored(const mtc::Graph &_graph, 
									thrust::device_vector<int> & dmatch,
									thrust::device_vector<int> & dfll,
									thrust::device_vector<int> & dbll,
									int iter){

        match = dmatch;
        fll = dfll;
        bll = dbll;										
		createColoredInputGraphViz(match, _graph, fll, bll);
		inputGraph->WriteToFile("inputGraph_iter_" + SSTR(iter));
		std::cout << "Wrote graph viz " << "iter_" + SSTR(iter) << std::endl;
}

void GraphViz::DrawSearchTree(int sizeOfSearchTree,
							int2 * searchTree,
							int iter){
	createSearchTreeGraphViz(sizeOfSearchTree, searchTree);
	searchTreeGraph->WriteToFile("searchTree_iter_" + SSTR(iter));
	std::cout << "Wrote graph viz " << "searchTree_iter_" + SSTR(iter) << std::endl;
}

void GraphViz::createSearchTreeGraphViz(int sizeOfSearchTree,
										int2 * searchTree){
	std::stringstream currSS;
	std::stringstream nextSS;
	for (int i = 0; i < sizeOfSearchTree; ++i){
		std::string node1Name;
		int2 currNode = searchTree[i];
		if (i == 0)
			node1Name = "root";
		else if (currNode.x == 0 && currNode.y == 0){
			printf("Current Leaf %d of search tree is null\n", i);
			continue;
		} else {
			currSS.str(std::string());
			currSS.clear();
			currSS << currNode.x << " " << currNode.y;
			node1Name = currSS.str();
		}
		nodeIt1 = searchTreeNodeMap.find(node1Name);
		if(nodeIt1 == searchTreeNodeMap.end()){
			searchTreeNodeMap[node1Name] = searchtreesubgraph->AddNode(node1Name);
			//searchTreeNodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(match[curr]));
			//searchTreeNodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[curr]));
			searchTreeNodeMap[node1Name]->GetAttributes().SetStyle("filled");
		}

		for (int c = 1; c <= 3; ++c){

			int2 childNode = searchTree[i*3 + c];
			if (childNode.x == 0 && childNode.y == 0){
				printf("Child Leaf %d of parent %d search tree is null\n", i*3 + c, i);
				continue;
			} else {
				printf("Child Leaf %d (%d, %d) of parent %d search tree is nonnull\n", i*3 + c,childNode.x,childNode.y, i);

			}
			nextSS.str(std::string());
			nextSS.clear();
			nextSS << childNode.x << " " << childNode.y;
			std::string node2Name = nextSS.str();

			nodeIt2 = searchTreeNodeMap.find(node2Name);
			if(nodeIt2 == searchTreeNodeMap.end()){
				searchTreeNodeMap[node2Name] = searchtreesubgraph->AddNode(node2Name);
				//searchTreeNodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(match[next]));
				//searchTreeNodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[next]));
				searchTreeNodeMap[node2Name]->GetAttributes().SetStyle("filled");
			}
			nodeIt1 = searchTreeNodeMap.find(node1Name);
			nodeIt2 = searchTreeNodeMap.find(node2Name);

			if(nodeIt1 != searchTreeNodeMap.end() && nodeIt2 != searchTreeNodeMap.end()) 
				searchtreesubgraph->AddEdge(searchTreeNodeMap[node1Name], searchTreeNodeMap[node2Name]); 
		}
	}
}

void GraphViz::createColoredInputGraphViz(thrust::host_vector<int> & match, 
					const mtc::Graph & g,
					thrust::host_vector<int> & fll,
					thrust::host_vector<int> & bll)
{
    for (int i = 0; i < g.nrVertices; ++i){
		// skip singletons
		if (fll[i] == i && bll[i] == i)
			continue;
		// Start from degrees only
		if (bll[i] == i){
			curr = i;
			next = fll[curr];
			while(curr != next){
				std::string node1Name = SSTR(curr);
				nodeIt1 = linearForestNodeMap.find(node1Name);
				if(nodeIt1 == linearForestNodeMap.end()){
					linearForestNodeMap[node1Name] = linearforestgraph->AddNode(node1Name);
					linearForestNodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(match[curr]));
					linearForestNodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[curr]));
					linearForestNodeMap[node1Name]->GetAttributes().SetStyle("filled");
				}
				std::string node2Name = SSTR(next);
				nodeIt2 = linearForestNodeMap.find(node2Name);
				if(nodeIt2 == linearForestNodeMap.end()){
					linearForestNodeMap[node2Name] = linearforestgraph->AddNode(node2Name);
					linearForestNodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(match[next]));
					linearForestNodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[next]));
					linearForestNodeMap[node2Name]->GetAttributes().SetStyle("filled");
				}
				nodeIt1 = linearForestNodeMap.find(node1Name);
				nodeIt2 = linearForestNodeMap.find(node2Name);

				if(nodeIt1 != linearForestNodeMap.end() && nodeIt2 != linearForestNodeMap.end()) 
					linearforestgraph->AddEdge(linearForestNodeMap[node1Name], linearForestNodeMap[node2Name]); 

				curr = next; 
				next = fll[curr];
			}
		}
	}

    // Since the graph doesnt grow uniformly, it is too difficult to only copy the new parts..
    for (int i = 0; i < g.nrVertices; ++i){
		std::string node1Name = SSTR(i);
        std::map<std::string, DotWriter::Node *>::const_iterator nodeIt1 = fullGraphNodeMap.find(node1Name);
        if(nodeIt1 == fullGraphNodeMap.end()) {
            fullGraphNodeMap[node1Name] = fullgraph->AddNode(node1Name);
			// Only color the linear forest vertices
            if(linearForestNodeMap.find(node1Name) != linearForestNodeMap.end()){
                fullGraphNodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(match[i]));
                fullGraphNodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[i]));
                fullGraphNodeMap[node1Name]->GetAttributes().SetStyle("filled");
            }
        }
        for (int j = g.neighbourRanges[i].x; j < g.neighbourRanges[i].y; ++j){
            //if (i < g.neighbours[j]){
                std::string node2Name = SSTR(g.neighbours[j]);
                std::map<std::string, DotWriter::Node *>::const_iterator nodeIt2 = fullGraphNodeMap.find(node2Name);
                if(nodeIt2 == fullGraphNodeMap.end()) {
                    fullGraphNodeMap[node2Name] = fullgraph->AddNode(node2Name);
                    if(linearForestNodeMap.find(node2Name) != linearForestNodeMap.end()){
                        fullGraphNodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(match[g.neighbours[j]]));
                        fullGraphNodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[g.neighbours[j]]));
                        fullGraphNodeMap[node2Name]->GetAttributes().SetStyle("filled");
                    }
                }  
                fullgraph->AddEdge(fullGraphNodeMap[node1Name], fullGraphNodeMap[node2Name]); 
            //}
        }
    }
}