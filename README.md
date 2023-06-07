How to run:

Train your input model:

python .\code\Context-LSTM\train_addittional_feats_py3.py -i <input_file> -d <directory> -s <separator> -n <num_add_feats>

Evaluate the suffix and remaining time prediction:

python .\code\Context-LSTM\evaluate_suffix_and_remaining_time_features_py3.py -i <input_file> -d <directory> -m <model path> -n <num_features>

TODO:
- Fix paths inside files: get a main reference because the file creating is wonky