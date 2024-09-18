#bash
echo $pwd
# download a zip from https://github.com/time-series-foundation-models/lag-llama/archive/daa31036bc4457dc7f8d5bcbe910f94dbbf3a30e.zip to the current tmp folder, unzip and rename it
current_dir=$(pwd)

rm -rf $current_dir/lag-llama
git clone https://github.com/time-series-foundation-models/lag-llama.git $current_dir/lag-llama
# cd
# install the requirements for lag-llama
pip install -r $current_dir/lag-llama/requirements.txt

# # install the lag-llama package
cd $current_dir/lag-llama
pip install .

# pip list | grep lag-llama
# wget https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt
wget https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt
# mv $current_dir/lag-llama.ckpt $current_dir/lag-llama

pip install "dask[dataframe]"
pip install pandas
