#include <iostream>
#include "gtest/gtest.h"

#include "addressUtils.h"

TEST(DeepgreenMatrix, BasicConstructor) {
  unsigned int depthOfSearchTree = 17;
  long long sizeOfSearchTree = CalculateSpaceForDesiredNumberOfLevels(depthOfSearchTree);
  std::vector<unsigned int> searchTree(sizeOfSearchTree);
  RecursivelyCallFillTree(0, 0, sizeOfSearchTree, &searchTree[0], "Start");
  for (long long start = 0; start < sizeOfSearchTree; ++start){
    if(start!= searchTree[start]){
      std::cout << "start "<< start <<std::endl;
      exit(1);
     }
      EXPECT_EQ(start, searchTree[start]);
  }

}
