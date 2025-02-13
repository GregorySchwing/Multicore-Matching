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
#ifndef MATCH_MATCH_H
#define MATCH_MATCH_H

#include <vector>

#include "graph.h"

#define NR_MATCH_ROUNDS 20
#define NR_MAX_MATCH_ROUNDS 256
//#define MATCH_INTERMEDIATE_COUNT 0
#define UNCOARSEN_GRAPH 1

namespace mtc
{

class GraphMatching
{
	public:
		GraphMatching(const Graph &);
		virtual ~GraphMatching();

		static void getWeight(double &, long &, const std::vector<int> &, const Graph &);
		static void getWeightGeneral(double &, std::vector<long> &, const std::vector<int> &, const std::vector<int> &, const std::vector<int> &, const Graph &);
		static bool testMatching(const std::vector<int> &, const Graph &);
		
		static inline int matchVal(const int &i, const int &j) {return 4 + (i < j ? i : j);};
		static inline bool isMatched(const int &m) {return m >= 4;};
		static inline bool isHead(const int &m, const std::vector<int> & bll) {return bll[m] == m;};

		void initialMatching(std::vector<int> & match);
		virtual void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution, int * kernelSolution, int leafIndex) const = 0;
		std::vector<int> match;

	protected:
		const Graph &graph;
};

class GraphMatchingCPURandom : public GraphMatching
{
	public:
		GraphMatchingCPURandom(const Graph &);
		~GraphMatchingCPURandom();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

class GraphMatchingCPUMinDeg : public GraphMatching
{
	public:
		GraphMatchingCPUMinDeg(const Graph &);
		~GraphMatchingCPUMinDeg();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

class GraphMatchingCPUStatMinDeg : public GraphMatching
{
	public:
		GraphMatchingCPUStatMinDeg(const Graph &);
		~GraphMatchingCPUStatMinDeg();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

class GraphMatchingCPUKarpSipser : public GraphMatching
{
	public:
		GraphMatchingCPUKarpSipser(const Graph &);
		~GraphMatchingCPUKarpSipser();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

class GraphMatchingCPUWeighted : public GraphMatching
{
	public:
		GraphMatchingCPUWeighted(const Graph &);
		~GraphMatchingCPUWeighted();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

class GraphMatchingCPUWeightedEdge : public GraphMatching
{
	public:
		GraphMatchingCPUWeightedEdge(const Graph &);
		~GraphMatchingCPUWeightedEdge();
		
		void performMatching(int *, cudaEvent_t &, cudaEvent_t &, int * dforwardlinkedlist,  int * dbackwardlinkedlist, int * dlength, int2 * dsearchtree, int * dynamicallyAddedVertices, int * numberOfDynamicallyAddedVertices, int sizeOfKernelSolution,  int * kernelSolution, int leafIndex) const;
};

};

#endif
