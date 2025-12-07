#!/bin/bash
# Pre-download SentenceTransformer model to avoid runtime errors
python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
