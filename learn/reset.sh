rm -rf model/* 
rm -rf data/*.json
rm -rf count*.json
python3 single_network.py
rm history.csv
rm log.log
cp ../ai/build/cpp_tic_tac_toe ./
