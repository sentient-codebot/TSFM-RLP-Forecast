#bash
echo "Current directory: $(pwd)"

# Define the current working directory
current_dir=$(pwd)

# Check if the lag-llama directory exists
if [ ! -d "$current_dir/lag-llama" ]; then
  echo "lag-llama directory not found. Cloning repository..."

  # Clone the lag-llama repository
  git clone https://github.com/time-series-foundation-models/lag-llama.git $current_dir/lag-llama

  # install the requirements for lag-llama
  pip install -r $current_dir/lag-llama/requirements.txt

  # # install the lag-llama package
  cd $current_dir/lag-llama
  pip install .

  # pip list | grep lag-llama
  wget https://huggingface.co/time-series-foundation-models/Lag-Llama/resolve/main/lag-llama.ckpt
  # mv $current_dir/lag-llama.ckpt $current_dir/lag-llama
else
  echo "lag-llama directory already exists. Skipping."
fi

# Install additional Python dependencies
pip install "dask[dataframe]" pandas
