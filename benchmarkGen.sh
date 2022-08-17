#!/bin/sh
#
# Script which generates all benchmark data for random and weighted matching.
# Uses the graphs listed by stats.sh.
# 
# Usage: ./benchmark.sh
#
device=1;
repeats=32;
for i in {1..5}
do
    randomfile="results/random$i.dat";
    rm -f $randomfile
    cat results/graphs.txt | sort -n | cut -f 2 | xargs -n 1 build/bin/match -r -d $device -a $repeats --gnuplot $randomfile -m 10
done



