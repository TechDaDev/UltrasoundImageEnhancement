#!/bin/bash

# Activate conda base environment and run Streamlit app
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate base
streamlit run app.py --server.fileWatcherType none --server.port 8501 --server.headless true
