#include <iostream>
#include "gtest/gtest.h"

#include "addressUtils.h"

TEST(DeepgreenMatrix, BasicConstructor) {
  int depthOfSearchTree = 13;
  long long sizeOfSearchTree = CalculateSpaceForDesiredNumberOfLevels(depthOfSearchTree);
  std::vector<int> searchTree(sizeOfSearchTree);
  RecursivelyCallFillTree(0, sizeOfSearchTree, &searchTree[0]);
  for (long long start = 0; start < sizeOfSearchTree; ++start){
      EXPECT_EQ(start, searchTree[start]);
  }

}
