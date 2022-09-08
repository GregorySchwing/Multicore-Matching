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

#define SSTR( x ) std::to_string( x )
//#define SSTR( x ) static_cast< std::ostringstream & >( \
//        ( std::ostringstream() << std::dec << x ) ).str()

GraphViz::GraphViz(){
	inputGraph = new DotWriter::RootGraph(false, "graph");
    linearforestgraph = inputGraph->AddSubgraph(subgraph1);
    fullgraph = inputGraph->AddSubgraph(subgraph2);

	searchTreeGraph = new DotWriter::RootGraph(false, "graph");
    searchtreesubgraph = searchTreeGraph->AddSubgraph(subgraph3);
}

void GraphViz::DrawInputGraphColored(const mtc::Graph &_graph, 
									int leafIndex,
									std::vector<int2> & searchtree,
									int UBDyn,
									std::vector<int> & dynamicallyaddedvertices,
									thrust::device_vector<int> & dmatch,
									thrust::device_vector<int> & dfll,
									thrust::device_vector<int> & dbll,
									int iter){

        match = dmatch;
        fll = dfll;
        bll = dbll;		
		linearForestNodeMap.clear();	   
        fullGraphNodeMap.clear();	
		inputGraph->RemoveSubgraph(linearforestgraph);
    	inputGraph->RemoveSubgraph(fullgraph);	
		linearforestgraph = inputGraph->AddSubgraph(subgraph1);
    	fullgraph = inputGraph->AddSubgraph(subgraph2);				
		createColoredInputGraphViz(match, leafIndex, searchtree, UBDyn, dynamicallyaddedvertices, _graph, fll, bll);
		inputGraph->WriteToFile("inputGraph_iter_" + SSTR(iter));
		std::cout << "Wrote graph viz " << "inputGraph_iter_" + SSTR(iter) << std::endl;
}

void GraphViz::DrawSearchTree(int sizeOfSearchTree,
							int2 * searchTree,
							int iter){
	createSearchTreeGraphViz(sizeOfSearchTree, searchTree);
	searchTreeGraph->WriteToFile("searchTree_iter_" + SSTR(iter));
	std::cout << "Wrote graph viz " << "searchTree_iter_" + SSTR(iter) << std::endl;
}

void GraphViz::DrawSearchTree(int sizeOfSearchTree,
							int2 * searchTree,
							std::string prefix){
	createSearchTreeGraphViz(sizeOfSearchTree, searchTree);
	searchTreeGraph->WriteToFile("searchTree_iter_" + prefix);
	std::cout << "Wrote graph viz " << "searchTree_iter_" + prefix << std::endl;
}

void GraphViz::createSearchTreeGraphViz(int sizeOfSearchTree,
										int2 * searchTree){
	std::stringstream currSS;
	std::stringstream nextSS;
	for (int i = 0; i < sizeOfSearchTree; ++i){

		std::string node1Name;
		std::string node1ID;
		int2 currNode = searchTree[i];
		if (i == 0){
			node1Name = "root";
			node1ID = SSTR(i);
		} else if (currNode.x == 0 && currNode.y == 0){
			continue;
		} else {
			currSS.str(std::string());
			currSS.clear();
			currSS << currNode.x << " " << currNode.y;
			node1Name = currSS.str();
			node1ID = SSTR(i);
		}
		nodeIt1 = searchTreeNodeMap.find(node1ID);
		if(nodeIt1 == searchTreeNodeMap.end()){
			searchTreeNodeMap[node1ID] = searchtreesubgraph->AddNode(node1Name, node1ID);
			//searchTreeNodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(match[curr]));
			//searchTreeNodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[curr]));
			searchTreeNodeMap[node1ID]->GetAttributes().SetStyle("filled");
		}

		for (int c = 1; c <= 3; ++c){
			if ((i*3 + c) >= sizeOfSearchTree)
				continue;
			int2 childNode = searchTree[i*3 + c];
			if (childNode.x == 0 && childNode.y == 0){
				continue;
			} else {
			}
			nextSS.str(std::string());
			nextSS.clear();
			nextSS << childNode.x << " " << childNode.y;
			std::string node2Name = nextSS.str();
			std::string node2ID = SSTR(i*3 + c);
			nodeIt2 = searchTreeNodeMap.find(node2ID);
			if(nodeIt2 == searchTreeNodeMap.end()){
				searchTreeNodeMap[node2ID] = searchtreesubgraph->AddNode(node2Name, node2ID);
				//searchTreeNodeMap[node2Name]->GetAttributes().SetColor(DotWriter::Color::e(match[next]));
				//searchTreeNodeMap[node2Name]->GetAttributes().SetFillColor(DotWriter::Color::e(match[next]));
				searchTreeNodeMap[node2ID]->GetAttributes().SetStyle("filled");
				searchtreesubgraph->AddEdge(searchTreeNodeMap[node1ID], searchTreeNodeMap[node2ID]); 
			} else {
				continue;
			}
		}
	}
}

void GraphViz::createColoredInputGraphViz(thrust::host_vector<int> & match, 
					int leafIndex,
					std::vector<int2> & searchtree,
					int UBDyn,
					std::vector<int> & dynamicallyaddedvertices,
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

	std::vector<int> soln;
	int leafIndexSoln = leafIndex;
	int2 nodeEntry;
	while(leafIndexSoln != 0){
		nodeEntry = searchtree[leafIndexSoln];
		soln.push_back(nodeEntry.x);
		soln.push_back(nodeEntry.y);
		if(leafIndexSoln % 3 == 0){
			--leafIndexSoln;
			leafIndexSoln = leafIndexSoln / 3;
		} else {
			leafIndexSoln = leafIndexSoln / 3;
		}
	}
	printf("Tree soln\n");
    for (int i = 0; i < soln.size(); ++i)
        printf("%d ", soln[i]);
    printf("\n");
	printf("Dyn soln\n");
    for (int i = 0; i < UBDyn; ++i)
        printf("%d ", dynamicallyaddedvertices[i]);
    printf("\n");

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
		if ((std::find(soln.begin(), soln.end(), i)) != soln.end()){
			fullGraphNodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(23));
			fullGraphNodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(23));
			fullGraphNodeMap[node1Name]->GetAttributes().SetStyle("filled");				
		}
		if ((std::find(dynamicallyaddedvertices.begin(), dynamicallyaddedvertices.begin() + UBDyn, i)) != dynamicallyaddedvertices.begin() + UBDyn){
			fullGraphNodeMap[node1Name]->GetAttributes().SetColor(DotWriter::Color::e(23));
			fullGraphNodeMap[node1Name]->GetAttributes().SetFillColor(DotWriter::Color::e(23));
			fullGraphNodeMap[node1Name]->GetAttributes().SetStyle("filled");				
		}
    }
}

void GraphViz::createColoredInputGraphViz(int * match, 
					int leafIndex,
					std::vector<int2> & searchtree,
					int UBDyn,
					std::vector<int> & dynamicallyaddedvertices,
					const mtc::Graph & g,
					int * fll,
					int * bll)
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