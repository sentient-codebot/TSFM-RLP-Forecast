#bash
echo $pwd
# download a zip from https://github.com/time-series-foundation-models/lag-llama/archive/daa31036bc4457dc7f8d5bcbe910f94dbbf3a30e.zip to the current tmp folder, unzip and rename it
current_dir=$(pwd)

# rm -rf $current_dir/lag-llama
# git clone https://github.com/time-series-foundation-models/lag-llama.git $current_dir/lag-llama
# cd
# install the requirements for lag-llama
pip install -r $current_dir/lag-llama/requirements.txt
# download pretrained model weights
# huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir $current_dir/lag-llama
# install the lag-llama package
pip install -e $current_dir/lag-llama
# pip list | grep lag-llama
pip install "dask[dataframe]"
