#!/bin/sh

for i in `seq 0 31`
do
    ./cpp_tic_tac_toe 1000
    rm -rf data/resolved*.json
    python3 train.py $i
    rm -rf data/selfplay*.json
done
