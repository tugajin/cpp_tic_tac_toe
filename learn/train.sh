#!/bin/sh

for i in `seq 0 100000`
do
    ./cpp_tic_tac_toe 1000
    python3 train.py $i
    rm -rf data/selfplay*.json
done
