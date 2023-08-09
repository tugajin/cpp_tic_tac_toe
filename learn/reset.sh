rm -rf model/*.h5
rm -rf model/*.save
rm -r model/*.pt 
rm -rf data/selfplay*.json
rm -rf count*.json
python3 single_network.py
rm history.csv
rm log.log
cp ../ai/build/cpp_tic_tac_toe ./
