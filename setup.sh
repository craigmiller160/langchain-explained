#!/bin/bash

python3 -m pip install langchain
python3 -m pip install langchain-community
python3 -m pip install chromadb
python3 -m pip install BeautifulSoup4
#docker compose up -d
#docker exec -it ollama ollama run llama2

echo ""
echo "*************************************************************************"
echo "* Everything is setup, but you need to run ollama/llama2 on your server *"
echo "*************************************************************************"