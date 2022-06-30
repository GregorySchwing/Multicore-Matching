#include <math.h>       /* log */

long long CalculateSpaceForDesiredNumberOfLevels(int levels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i <= levels; ++i){
        summand += pow (3.0, i);
        finishedLeavesPerLevel[i] = 0;
        totalLeavesPerLevel[i] = pow (3.0, i);
    }
    return summand;
}

void RecursivelyCallFillTree(int leafIndex,
                            int * searchTree){
    // Random number between 1 and N
    int numLeavesToFill = rand();
    std::array<int, 4> newLeaves = CalculateLeafOffsets(leafIndex,numLeavesToFill);
    int lb = newLeaves[0];
    int ub = newLeaves[1];
    while(lb < ub && lb < sizeOfSearchTree){
        RecursivelyCallFillTree(lb, searchTree);
        ++lb;
    }
    depthOfLeaf = ceil(logf(2*newLeaves.w + 1) / logf(3)) - 1;
    lb = newLeaves[2];
    ub = newLeaves[3];
    while(lb < ub && lb < sizeOfSearchTree){
        RecursivelyCallFillTree(lb, searchTree);
        ++lb;
    }
}

void FillTree(int leafIndex,
              int fullpathcount,
              int * searchTree){
    int leavesToProcess = fullpathcount;
    int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3));
    int arbitraryParameter = 3*((3*leafIndex)+1);
    int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, incompleteLevel-1) - 3)/6;

    int leavesFromIncompleteLevelLvl = powf(3.0, incompleteLevel); 
    int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, (incompleteLevel-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons???
    int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

    if (levelOffset + 0 >= sizeOfSearchTree){
        //printf("child %d exceeded srch tree depth\n", levelOffset);
        return;        
    }

    searchTree[levelOffset + 0] = levelOffset + 0;
    searchTree[levelOffset + 1] = levelOffset + 1;
    searchTree[levelOffset + 2] = levelOffset + 2;
}

// Template this to do any type of tree
// binary, ternary, quaternary, ...
std::array<int, 4> CalculateLeafOffsets(int leafIndex,
                                        int fullpathcount){
    int arbitraryParameter;
    int leftMostLeafIndexOfFullLevel;
    int leftMostLeafIndexOfIncompleteLevel;
    int leavesToProcess = fullpathcount;
    
    if (leavesToProcess == 0){
        int arr[] = {
            leafIndex,
            leafIndex,
            leafIndex,
            leafIndex
        };
        std::array<int,4> myarray;
        std::copy(std::begin(arr), std::end(arr), myarray.begin());
        return myarray;
    }
    
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for leavesToProcess < closed form
    // start from level 1, hence add a level if LTP > 0, 1 complete level 
    // Add 1 if LTP == 0 to prevent runtime error
    // LTP = 2
    // CL = 1
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    int completeLevel = floor(logf(2*leavesToProcess + 1) / logf(3));
    // If LTP == 0, we dont want to create any new leaves
    // Therefore, we dont want to enter the for loops.
    // The active leaf writes itself as it's parent before the for loops
    // This is overwritten within the for loops if LTP > 0
    // CLL = 3
    int leavesFromCompleteLvl = powf(3.0, completeLevel);
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for closed form < leavesToProcess
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    // IL = 1
    int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3));
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Add 1 when leavesToProcess isn't 0, so we start counting from level 1
    // Also subtract the root, so we start counting from level 1
    // TSC = 3
    int treeSizeComplete = (1.0 - powf(3.0, completeLevel+1))/(1.0 - 3.0);
    // How many internal leaves to skip in complete level
    // RFC = 1
    int removeFromComplete = ((3*leavesToProcess - treeSizeComplete) + 3 - 1) / 3;
    // Leaves that are used in next level
    int leavesFromIncompleteLvl = 3*removeFromComplete;
    
    // Test from root for now, this code can have an arbitrary root though
    arbitraryParameter = 3*((3*leafIndex)+1);
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    leftMostLeafIndexOfFullLevel = ((2*arbitraryParameter+3)*powf(3.0, completeLevel-1) - 3)/6;
    leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, incompleteLevel-1) - 3)/6;

    int totalNewActive = (leavesFromCompleteLvl - removeFromComplete) + leavesFromIncompleteLvl;
    #ifndef NDEBUG
    printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, completeLevel);
    printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, incompleteLevel);
    printf("Leaves %d, treeSizeComplete %d\n",leavesToProcess, treeSizeComplete);
    printf("Leaves %d, totalNewActive %d\n",leavesToProcess, totalNewActive);
    printf("Leaves %d, leavesFromCompleteLvl %d\n",leavesToProcess, leavesFromCompleteLvl);
    printf("Leaves %d, leavesFromIncompleteLvl %d\n",leavesToProcess, leavesFromIncompleteLvl);
    printf("Leaves %d, leftMostLeafIndexOfFullLevel %d\n",leavesToProcess, leftMostLeafIndexOfFullLevel);
    printf("Leaves %d, leftMostLeafIndexOfIncompleteLevel %d\n",leavesToProcess, leftMostLeafIndexOfIncompleteLevel);
    #endif
    // Grow tree leftmost first, so put the incomplete level first.
    // Shape of leaves
    //CL    -     -    o o o 
    //IL  o o o o o o
    

    int arr[] = {
        leftMostLeafIndexOfIncompleteLevel,
        leftMostLeafIndexOfIncompleteLevel + leavesFromIncompleteLvl,
        leftMostLeafIndexOfFullLevel,
        leftMostLeafIndexOfFullLevel + leavesFromCompleteLvl
    };
    std::array<int,4> myarray;
    std::copy(std::begin(arr), std::end(arr), myarray.begin());
    return myarray;

}




// Template this to do any type of tree
// binary, ternary, quaternary, ...
std::array<int, 3> PopulateTreeArithmetic(int leafIndex,
                                        int & fullpathcount){


    // post increment is equivalent to atomicAdd
    int leavesToProcess = fullpathcount++;
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for leavesToProcess < closed form
    // start from level 1, hence add a level if LTP > 0, 1 complete level 
    // Add 1 if LTP == 0 to prevent runtime error
    // LTP = 2
    // CL = 1
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3));
    int arbitraryParameter = 3*((3*leafIndex)+1);
    int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, incompleteLevel-1) - 3)/6;

    int leavesFromIncompleteLevelLvl = powf(3.0, incompleteLevel); 
    int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, (incompleteLevel-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;
    #ifndef NDEBUG
    printf("Level Depth %d\n", incompleteLevel);
    printf("Level Width  %d\n", leavesFromIncompleteLevelLvl);
    printf("Size of Tree %d\n", treeSizeNotIncludingThisLevel);
    printf("Global level left offset (GLLO) %d\n", leftMostLeafIndexOfIncompleteLevel);
    printf("internalLeafIndex %d\n", internalLeafIndex);
    printf("parent (%d)'s child indices %d %d %d \n", leafIndex, 
                                                levelOffset,
                                                levelOffset + 1,
                                                levelOffset + 2);
    #endif


    int arr[] = {levelOffset + 0, levelOffset + 1, levelOffset + 2};
    std::array<int,3> myarray;
    std::copy(std::begin(arr), std::end(arr), myarray.begin());
    return myarray;
}