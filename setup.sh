#!/bin/bash

python3 -m pip install langchain
python3 -m pip install langchain-community
docker compose up -d
docker exec -it ollama ollama run llama2