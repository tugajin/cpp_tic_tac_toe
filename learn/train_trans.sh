#!/bin/sh
while true
do
	./cpp_tic_tac_toe 1000 || exit 1
	python3 train_worker_trans.py || exit 1
done
