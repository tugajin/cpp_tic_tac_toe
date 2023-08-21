rm -rf model/*.h5
rm -rf model/*.save
rm -r model/*.pt 
rm -rf data/selfplay*.json
rm -rf data/resolved*.json
rm -rf data/const.json
rm -rf count*.json
#python3 generate_transformer_model.py
python3 generate_poolformer_model.py
rm history.csv
rm selfplay_result.csv
rm log.log
cp ../ai/build/cpp_tic_tac_toe ./
