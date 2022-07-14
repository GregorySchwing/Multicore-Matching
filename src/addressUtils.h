#include <math.h>       /* log */

std::array<unsigned int, 4> CalculateLeafOffsets(unsigned int leafIndex,
                                        unsigned int fullpathcount);


std::array<unsigned int, 4> CalculateLeafOffsets2(unsigned int leafIndex,
                                        unsigned int fullpathcount);                                        
void FillTree(unsigned int leafIndex,
              unsigned int & fullpathcount,
              unsigned int sizeOfSearchTree,
              unsigned int * searchTree);
long long CalculateSpaceForDesiredNumberOfLevels(unsigned int levels){
    long long summand= 0;
    // ceiling(vertexCount/2) loops
    for (unsigned int i = 0; i <= levels; ++i){
        summand += pow (3.0, i);
    }
    return summand;
}

void RecursivelyCallFillTree(unsigned int leafIndex,
                            unsigned int parentLeafIndex,
                            long long sizeOfSearchTree,
                            unsigned int * searchTree,
                            std::string label){
    if(leafIndex >= sizeOfSearchTree)
        return;
    //else
    //    printf("LI %d PLI %d\n", leafIndex, parentLeafIndex);
    //    std::cout << label << std::endl;

    // Random number between 1 and N
    unsigned int numLeavesToFill = 1+rand()%20;
    //unsigned int numLeavesToFill = sizeOfSearchTree;
    //unsigned int numLeavesToFill = 1;
    //std::array<unsigned int, 4> newLeaves = CalculateLeafOffsets(leafIndex,numLeavesToFill);



    //std::cout << label << std::endl;

    FillTree(   leafIndex, 
                numLeavesToFill,
                sizeOfSearchTree,
                searchTree);

    std::array<unsigned int, 4> newLeaves = CalculateLeafOffsets2(leafIndex,numLeavesToFill);

    unsigned int ilb = newLeaves[0];
    unsigned int iub = newLeaves[1];
    unsigned int clb = newLeaves[2];
    unsigned int cub = newLeaves[3];

    while(ilb < iub && ilb < sizeOfSearchTree){
        RecursivelyCallFillTree(ilb, leafIndex, sizeOfSearchTree, searchTree, "incomplete");
        ++ilb;
    }
    //unsigned int depthOfLeaf = ceil(logf(2*newLeaves.w + 1) / logf(3)) - 1;
    // %d CL LB %d UB %d\n",leafIndex,numLeavesToFill,lb, ub);
    while(clb < cub && clb < sizeOfSearchTree){
        RecursivelyCallFillTree(clb, leafIndex, sizeOfSearchTree, searchTree, "complete");
        ++clb;
    }

}

// This method depends on being able to change the number of fullpathcounts.
// This can be accomplished by either an atomic locking variable
// or possible some logic to handle the leaf nodes.
void FillTree(unsigned int leafIndex,
              unsigned int & fullpathcount,
              unsigned int sizeOfSearchTree,
              unsigned int * searchTree){

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
    for (unsigned int leavesToProcess = 1; leavesToProcess < fullpathcount+1; ++leavesToProcess){
        unsigned int n = ceil(logf(2*leavesToProcess + 1) / logf(3));
        //float nf = logf(2*leavesToProcess + 1) / logf(3);
        unsigned int arbitraryParameter = 3*((3*leafIndex)+1);
        // At high powers, the error of transendental powf causes bugs.
        //unsigned int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

        // Discrete calculation without trancendentals
        unsigned int leftMostLeafIndexOfIncompleteLevel = (2*arbitraryParameter+3);
        unsigned int multiplicand = 1;
        for (unsigned int i = 1; i < n; ++i)
            multiplicand*=3;
        leftMostLeafIndexOfIncompleteLevel*=multiplicand;
        leftMostLeafIndexOfIncompleteLevel-=3;
        leftMostLeafIndexOfIncompleteLevel/=6;

        unsigned int treeSizeNotIncludingThisLevel = (1.0 - multiplicand)/(1.0 - 3.0); 
        // At high powers, the error of transendental powf causes bugs. 
        //unsigned int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, ((n+1)-1)))/(1.0 - 3.0);  
        // Test from root for now, this code can have an arbitrary root though
        //leafIndex = global_active_leaves[globalIndex];
    //    leafIndex = 0;
        // Closed form solution of recurrence relation shown in comment above method
        // Subtract 1 because reasons???
        unsigned int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
        //unsigned int internalLeafIndex = leavesToProcess - treeSizeNotIncludingThisLevel;
        unsigned int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

        if (sizeOfSearchTree <= levelOffset ||
        sizeOfSearchTree <= (levelOffset + 1) ||
        sizeOfSearchTree <= (levelOffset +2)||
        levelOffset< 0 ||
        (levelOffset+1) < 0 ||
        (levelOffset + 2)< 0){
            fullpathcount = leavesToProcess;
            return;
        }
        if (searchTree[levelOffset] != 0){
        printf("Leaves %d, leafIndex %d\n",leavesToProcess, leafIndex);
        printf("Leaves %d, multiplicand %d\n",leavesToProcess, multiplicand);
        printf("Leaves %d, n %d\n",leavesToProcess, n);

        printf("Leaves %d, internalLeafIndex %d\n",leavesToProcess, internalLeafIndex);
        printf("Leaves %d, leftMostLeafIndexOfIncompleteLevel %d\n",leavesToProcess, leftMostLeafIndexOfIncompleteLevel);
        printf("Leaves %d, treeSizeNotIncludingThisLevel %d\n",leavesToProcess, treeSizeNotIncludingThisLevel);
        printf("Leaves %d, n %d\n",leavesToProcess, n);
                    exit(0);

        }
        if (searchTree[levelOffset + 1] != 0){
        printf("Leaves %d, leafIndex %d\n",leavesToProcess, leafIndex);
        printf("Leaves %d, internalLeafIndex %d\n",leavesToProcess, internalLeafIndex);
        printf("Leaves %d, leftMostLeafIndexOfIncompleteLevel %d\n",leavesToProcess, leftMostLeafIndexOfIncompleteLevel);
        printf("Leaves %d, treeSizeNotIncludingThisLevel %d\n",leavesToProcess, treeSizeNotIncludingThisLevel);
        printf("Leaves %d, n %d\n",leavesToProcess, n);
        }
        if (searchTree[levelOffset + 2] != 0){
        printf("Leaves %d, leafIndex %d\n",leavesToProcess, leafIndex);
        printf("Leaves %d, internalLeafIndex %d\n",leavesToProcess, internalLeafIndex);
        printf("Leaves %d, leftMostLeafIndexOfIncompleteLevel %d\n",leavesToProcess, leftMostLeafIndexOfIncompleteLevel);
        printf("Leaves %d, treeSizeNotIncludingThisLevel %d\n",leavesToProcess, treeSizeNotIncludingThisLevel);
        printf("Leaves %d, n %d\n",leavesToProcess, n);
        }

        searchTree[levelOffset + 0] += levelOffset + 0;
        searchTree[levelOffset + 1] += levelOffset + 1;
        searchTree[levelOffset + 2] += levelOffset + 2;

    }
}

// Template this to do any type of tree
// binary, ternary, quaternary, ...
std::array<unsigned int, 4> CalculateLeafOffsets(unsigned int leafIndex,
                                        unsigned int fullpathcount){
    unsigned int arbitraryParameter;
    unsigned int leftMostLeafIndexOfFullLevel;
    unsigned int leftMostLeafIndexOfIncompleteLevel;
    unsigned int leavesToProcess = fullpathcount;
    
    if (leavesToProcess == 0){
        unsigned int arr[] = {
            leafIndex,
            leafIndex,
            leafIndex,
            leafIndex
        };
        std::array<unsigned int,4> myarray;
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
    unsigned int completeLevel = floor(logf(2*leavesToProcess + 1) / logf(3));

    // At high powers, the error of transendental powf causes bugs.
    //unsigned int leavesFromCompleteLvlTest = powf(3.0, completeLevel);
    unsigned int leavesFromCompleteLvl = 1;
    for (unsigned int i = 1; i <= completeLevel; ++i)
        leavesFromCompleteLvl*=3;
   // if (leavesFromCompleteLvlTest != leavesFromCompleteLvl)  
    //    exit(1);

    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for closed form < leavesToProcess
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    // IL = 1
    unsigned int incompleteLevel = ceil(logf(2*leavesToProcess + 1) / logf(3));
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // At high powers, the error of transendental powf causes bugs.
    //unsigned int treeSizeCompleteTest = (1.0 - powf(3.0, completeLevel+1))/(1.0 - 3.0);
    unsigned int treeSizeComplete = (1.0 - 3*leavesFromCompleteLvl)/(1.0 - 3.0);
    //if (treeSizeCompleteTest != treeSizeComplete)  
    //    exit(1);
    // How many internal leaves to skip in complete level
    // RFC = 1
    unsigned int removeFromComplete = ((3*leavesToProcess - treeSizeComplete) + 3 - 1) / 3;
    // Leaves that are used in next level
    unsigned int leavesFromIncompleteLvl = 3*removeFromComplete;
    
    // Test from root for now, this code can have an arbitrary root though
    arbitraryParameter = 3*((3*leafIndex)+1);
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    unsigned int multiplicandIL = 1;
    for (unsigned int i = 1; i < incompleteLevel; ++i)
        multiplicandIL*=3;

    unsigned int multiplicandFL = 1;
    for (unsigned int i = 1; i < completeLevel; ++i)
        multiplicandFL*=3;        

    leftMostLeafIndexOfFullLevel = ((2*arbitraryParameter+3)*multiplicandFL - 3)/6;
    leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*multiplicandIL - 3)/6;

    unsigned int totalNewActive = (leavesFromCompleteLvl - removeFromComplete) + leavesFromIncompleteLvl;
    #ifndef NDEBUG
    printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, completeLevel);
    printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, incompleteLevel);
    printf("Leaves %d, treeSizeComplete %d\n",leavesToProcess, treeSizeComplete);
    printf("Leaves %d, removeFromComplete %d\n",leavesToProcess, removeFromComplete);
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

    unsigned int arr[] = {
        leftMostLeafIndexOfIncompleteLevel,
        leftMostLeafIndexOfIncompleteLevel + leavesFromIncompleteLvl,
        leftMostLeafIndexOfFullLevel + removeFromComplete,
        leftMostLeafIndexOfFullLevel + leavesFromCompleteLvl
    };
    std::array<unsigned int,4> myarray;
    std::copy(std::begin(arr), std::end(arr), myarray.begin());
    return myarray;

}

// Template this to do any type of tree
// binary, ternary, quaternary, ...
std::array<unsigned int, 4> CalculateLeafOffsets2(unsigned int leafIndex,
                                        unsigned int fullpathcount){

    unsigned int leavesToProcess = fullpathcount;
    unsigned int leavesFromIncompleteLvl = 1;
    unsigned int leavesFromCompleteLvl = 1;

    if (leavesToProcess == 0){
        unsigned int arr[] = {
            leafIndex,
            leafIndex,
            leafIndex,
            leafIndex
        };
        std::array<unsigned int,4> myarray;
        std::copy(std::begin(arr), std::end(arr), myarray.begin());
        return myarray;
    }
    unsigned int n_com = floor(logf(2*leavesToProcess + 1) / logf(3));
    unsigned int n_inc = ceil(logf(2*leavesToProcess + 1) / logf(3));

    for (unsigned int i = 1; i <= n_inc; ++i)
        leavesFromIncompleteLvl*=3;

    for (unsigned int i = 1; i <= n_com; ++i)
        leavesFromCompleteLvl*=3;
    //float nf = logf(2*leavesToProcess + 1) / logf(3);
    unsigned int arbitraryParameter = 3*((3*leafIndex)+1);
    // At high powers, the error of transendental powf causes bugs.
    //unsigned int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

    // Discrete calculation without trancendentals
    unsigned int leftMostLeafIndexOfIncompleteLevel = (2*arbitraryParameter+3);
    unsigned int multiplicandn_inc = 1;
    for (unsigned int i = 1; i < n_inc; ++i)
        multiplicandn_inc*=3;
    leftMostLeafIndexOfIncompleteLevel*=multiplicandn_inc;
    leftMostLeafIndexOfIncompleteLevel-=3;
    leftMostLeafIndexOfIncompleteLevel/=6;

    unsigned int leftMostLeafIndexOfCompleteLevel = (2*arbitraryParameter+3);
    unsigned int multiplicandn_com = 1;
    for (unsigned int i = 1; i < n_com; ++i)
        multiplicandn_com*=3;
    leftMostLeafIndexOfCompleteLevel*=multiplicandn_com;
    leftMostLeafIndexOfCompleteLevel-=3;
    leftMostLeafIndexOfCompleteLevel/=6;

    unsigned int treeSizeNotIncludingThisLevel = (1.0 - multiplicandn_inc)/(1.0 - 3.0); 
    // At high powers, the error of transendental powf causes bugs. 
    //unsigned int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, ((n+1)-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons???
    unsigned int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    //unsigned int internalLeafIndex = leavesToProcess - treeSizeNotIncludingThisLevel;
    unsigned int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;

    #ifndef NDEBUG
    printf("Leaves %d, completeLevel Level Depth %d\n",leavesToProcess, completeLevel);
    printf("Leaves %d, incompleteLevel Level Depth %d\n",leavesToProcess, incompleteLevel);
    printf("Leaves %d, treeSizeComplete %d\n",leavesToProcess, treeSizeComplete);
    printf("Leaves %d, removeFromComplete %d\n",leavesToProcess, removeFromComplete);
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
    unsigned int clb;
    unsigned int cub;

    if (n_com == n_inc){
        clb = levelOffset + 3;
        cub = levelOffset + 3;
    } else {
        clb = (levelOffset + 2)/3;
        cub = leftMostLeafIndexOfCompleteLevel + leavesFromCompleteLvl;
    }
    unsigned int arr[] = {
        leftMostLeafIndexOfIncompleteLevel,
        levelOffset + 3,
        clb,
        cub
    };
    std::array<unsigned int,4> myarray;
    std::copy(std::begin(arr), std::end(arr), myarray.begin());
    return myarray;

}


// Template this to do any type of tree
// binary, ternary, quaternary, ...
std::array<unsigned int, 3> PopulateTreeArithmetic(unsigned int leafIndex,
                                        unsigned int & fullpathcount){


    // post increment is equivalent to atomicAdd
    unsigned int leavesToProcess = fullpathcount++;
    // https://en.wikipedia.org/wiki/Geometric_series#Closed-form_formula
    // Solved for leavesToProcess < closed form
    // start from level 1, hence add a level if LTP > 0, 1 complete level 
    // Add 1 if LTP == 0 to prevent runtime error
    // LTP = 2
    // CL = 1
    // Always add 2 to prevent run time error, also to start counting at level 1 not level 0
    unsigned int n = ceil(logf(2*leavesToProcess + 1) / logf(3));
    unsigned int arbitraryParameter = 3*((3*leafIndex)+1);
    unsigned int leftMostLeafIndexOfIncompleteLevel = ((2*arbitraryParameter+3)*powf(3.0, n-1) - 3)/6;

    unsigned int treeSizeNotIncludingThisLevel = (1.0 - powf(3.0, (n-1)))/(1.0 - 3.0);  
    // Test from root for now, this code can have an arbitrary root though
    //leafIndex = global_active_leaves[globalIndex];
//    leafIndex = 0;
    // Closed form solution of recurrence relation shown in comment above method
    // Subtract 1 because reasons
    unsigned int internalLeafIndex = leavesToProcess - 1 - treeSizeNotIncludingThisLevel;
    unsigned int levelOffset = leftMostLeafIndexOfIncompleteLevel + 3*internalLeafIndex;
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


    unsigned int arr[] = {levelOffset + 0, levelOffset + 1, levelOffset + 2};
    std::array<unsigned int,3> myarray;
    std::copy(std::begin(arr), std::end(arr), myarray.begin());
    return myarray;
}