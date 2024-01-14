FROM --platform=linux/amd64 ollama/ollama:latest

#RUN echo "#!/bin/bash" > /entrypoint.sh
#RUN echo "ollama serve &" >> /entrypoint.sh
#RUN echo "ollama pull llama2" >> /entrypoint.sh
#RUN chmod 777 /entrypoint.sh

#ENTRYPOINT ["/entrypoint.sh"]