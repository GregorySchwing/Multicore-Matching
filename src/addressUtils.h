#include <math.h>       /* log */

std::array<int, 4> CalculateLeafOffsets(int leafIndex,
                                        int fullpathcount);
void FillTree(int leafIndex,
              int fullpathcount,
              int sizeOfSearchTree,
              int * searchTree);
long long CalculateSpaceForDesiredNumberOfLevels(int levels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (int i = 0; i <= levels; ++i){
        summand += pow (3.0, i);
    }
    return summand;
}

void RecursivelyCallFillTree(int leafIndex,
                            long long sizeOfSearchTree,
                            int * searchTree){
    if(leafIndex >= sizeOfSearchTree)
        return;
    // Random number between 1 and N
    int numLeavesToFill = 1+rand()%20;
    //int numLeavesToFill = sizeOfSearchTree;

    std::array<int, 4> newLeaves = CalculateLeafOffsets(leafIndex,numLeavesToFill);

    int ilb = newLeaves[0];
    int iub = newLeaves[1];
    int clb = newLeaves[2];
    int cub = newLeaves[3];
    //printf("Root %d new leaves %d IL LB %d UB %d\n",leafIndex,numLeavesToFill,ilb, iub);
    //printf("Root %d new leaves %d CL LB %d UB %d\n",leafIndex,numLeavesToFill,clb, cub);

    FillTree(   leafIndex, 
                numLeavesToFill,
                sizeOfSearchTree,
                searchTree);

    while(ilb < iub && ilb < sizeOfSearchTree){
        RecursivelyCallFillTree(ilb, sizeOfSearchTree, searchTree);
        ++ilb;
    }
    //int depthOfLeaf = ceil(logf(2*newLeaves.w + 1) / logf(3)) - 1;
    // %d CL LB %d UB %d\n",leafIndex,numLeavesToFill,lb, ub);
    while(clb < cub && clb < sizeOfSearchTree){
        RecursivelyCallFillTree(clb, sizeOfSearchTree, searchTree);
        ++clb;
    }

}

void FillTree(int leafIndex,
              int fullpathcount,
              int sizeOfSearchTree,
              int * searchTree){
    int leavesToProcess = fullpathcount;

    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // r = 3, a = 1, solve for n given s_n = leavesToProcess ∈ [1,m/4]
    // where m = number of vertices.
    // s_n = (1-r^(n+1))/(1-r)
    // s_n * (1-3) = -2*s_n = (1-r^(n+1))
    //     = -2*s_n - 1 = -3^(n+1)
    //     = 2*s_n + 1  =  3^(n+1)
    //     = log(2*s_n + 1) = n+1*log(3)
    //     = log(2*s_n + 1)/log(3) = n + 1
    //     = log(2*s_n + 1)/log(3) - 1 = n
    // n is the number of terms in the closed form solution.
    // Alternatively, n is the number of levels in the search tree.
    for (int leavesToProcess = 1; leavesToProcess < fullpathcount+1; ++leavesToProcess){
        int n = ceil(logf(2*leavesToProcess + 1) / logf(3));
        float nf = logf(2*leavesToProcess + 1) / logf(3);
        int arbitraryParameter = 3*((3*leafIndex)+1);
        // At high powers, the error of transendental powf causes bugs.
        //int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

        // Discrete calculation without trancendentals
        int leftMostLeafIndexOfIncompleteLevel = (2*arbitraryParameter+3);
        int multiplicand = 1;
        for (int i = 1; i < n; ++i)
            multiplicand*=3;
        leftMostLeafIndexOfIncompleteLevel*=multiplicand;
        leftMostLeafIndexOfIncompleteLevel-=3;
        leftMostLeafIndexOfIncompleteLevel/=6;

        int treeSizeNotIncludingThisLevel = (1.0 - multiplicand)/(1.0 - 3.0); 
        // At high powers, the error of transendental powf causes bugs. 
        //int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, ((n+1)-1)))/(1.0 - 3.0);  
        // Test from root for now, this code can have an arbitrary root though
        //leafIndex = global_active_leaves[globalIndex];
    //    leafIndex = 0;
        // Closed form solution of recurrence relation shown in comment above method
        // Subtract 1 because reasons???
        int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
        //int internalLeafIndex = leavesToProcess - treeSizeNotIncludingThisLevel;
        int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

        if (levelOffset + 0 >= sizeOfSearchTree){
            //printf("child %d exceeded srch tree depth\n", levelOffset);
            return;        
        }
        
        searchTree[levelOffset + 0] += levelOffset + 0;
        searchTree[levelOffset + 1] += levelOffset + 1;
        searchTree[levelOffset + 2] += levelOffset + 2;

    }
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
    int multiplicandIL = 1;
    for (int i = 1; i < incompleteLevel; ++i)
        multiplicandIL*=3;

    int multiplicandFL = 1;
    for (int i = 1; i < completeLevel; ++i)
        multiplicandFL*=3;        

    leftMostLeafIndexOfFullLevel = ((2*arbitraryParameter+3)*multiplicandFL - 3)/6;
    leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*multiplicandIL - 3)/6;

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
    int n = ceil(logf(2*leavesToProcess + 1) / logf(3));
    int arbitraryParameter = 3*((3*leafIndex)+1);
    int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

    int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, (n-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;
    #ifndef NDEBUG
    printf("Level Depth %d\n", n);
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