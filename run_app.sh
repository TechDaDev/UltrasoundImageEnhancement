#!/bin/bash

# Activate virtual environment and run Streamlit app
source venv/bin/activate
streamlit run app.py --server.fileWatcherType none --server.port 8501 --server.headless true
