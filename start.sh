#!/bin/bash
# Quick-start Streamlit on RunPod
python -m streamlit run app.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
