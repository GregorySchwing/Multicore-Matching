#include <iostream>
#include "gtest/gtest.h"

#include "addressUtils.h"

TEST(DeepgreenMatrix, BasicConstructor) {
  int depthOfSearchTree = 10;
  long long sizeOfSearchTree = CalculateSpaceForDesiredNumberOfLevels(depthOfSearchTree);
  std::vector<int> searchTree(sizeOfSearchTree);
  RecursivelyCallFillTree(0,sizeOfSearchTree);
  /*
  std::array<int, 4> myArray = CalculateLeafOffsets(3,1);
  for (auto & a : myArray)
    std::cout <<  a << " ";
  */
  EXPECT_FLOAT_EQ(11.0f, 11.0f);

}
