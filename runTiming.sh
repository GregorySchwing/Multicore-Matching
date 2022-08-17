#!/bin/bash
# Example with 28 cores for OpenMP
#
# Project/Account
#SBATCH -A greg
#
# Number of cores
#SBATCH -c 1 -w, --nodelist=potoff33
#
# Runtime of this jobs is less then 12 hours.
#SBATCH --time=2-0:00:00
#
#SBATCH --mail-user=go2432@wayne.edu

#SBATCH -o output_%j.out

#SBATCH -e errors_%j.err


# Clear the environment from any previously loaded modules
now=$(date)
echo "$now"
./build/bin/cover -m 10 SS-Butterfly_weights_shifted_1_10K_Edges_Row_Major.mtx
now=$(date)
echo "$now"
# End of submit file
