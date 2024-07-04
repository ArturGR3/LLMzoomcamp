
### Notes for HW2 
Start olama server
```bash start olama
ollama start  
```

Run ollama with phi3
```bash 
olama run phi3
```
Stop olama server
```bash
sudo systemctl stop ollama.service
```

[1] Create a docker container with ollama
```bash 
docker run -it \
    --rm \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```
[2] Get the version of ollama
```bash 
docker exec -it ollama ollama -v
``` 

Find metadata about this model
```bash
docker exec -it ollama cat /root/.ollama/models/manifests/registry.ollama.ai/library/gemma/2b
```

Run Ollama gemma:2b (10*10)
```bash
echo 10*10 | docker exec -i ollama ollama run gemma:2b
```

Download weight locally 

[1] Stop container 
```bash 
docker stop ollama
```

[2] Create a new container that stores weights locallaly
```bash 
docker run -it \
    --rm \
    -v ./ollama_files:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama

docker exec -it ollama ollama pull gemma:2b
```

Adding the weights to docker run
```Dockerfile
FROM ollama/ollama

# Copy the pre-downloaded weights from the local directory to the container
COPY ollama_files /root/.ollama
```

build a docker container with locally saved weights
```bash 
docker build -t ollama_with_weights .
docker run -it \
        --rm \
        -p 11434:11434 \
        --name ollama_with_weights \
        ollama_with_weights
```
run a prompt against ollama with temperature paramater
```bash 
curl http://localhost:11434/api/generate -d '{
  "model": "gemma:2b",
  "prompt": "Tell me a funny joke about soccer player",
  "temperature": 0.0
}' | jq -r '.response' | tr -d '\n'
```