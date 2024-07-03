### Start olama server
```bash
ollama start 
```

### Run ollama with phi3

```bash 
olama run phi3
```

### Stop olama server
```bash
sudo systemctl stop ollama.service
```

###  Questions 1. Create a docker container with olama and get the version of olama using exec  
[1] Create a docker container 
```bash 
docker run -it \
    --rm \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```
[2] Get the version of Ollama 
```bash 
docker exec -it ollama ollama -v
``` 
ollama version is 0.1.48

###  Questions 2. Find metadata about this model
```bash
docker exec -it ollama cat /root/.ollama/models/manifests/registry.ollama.ai/library/gemma/2b
```

### Question 3. Run Ollama gemma:2b (10*10)
```bash
echo 10*10 | docker exec -i ollama ollama run gemma:2b
```

### Question 4. Download weight locally 

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
### Question 5. Adding the weights
Create a Dockerfile: 

```Dockerfile
FROM ollama/ollama

# Copy the pre-downloaded weights from the local directory to the container
COPY ollama_files /root/.ollama
```

```bash 
docker build -t ollama_with_weights .
docker run -it \
        --rm \
        -p 11434:11434 \
        --name ollama_with_weights \
        ollama_with_weights
```

```bash 
curl http://localhost:11434/api/generate -d '{
  "model": "gemma:2b",
  "prompt": "Tell me a funny joke about soccer player",
  "temperature": 0.0
}' | jq -r '.response' | tr -d '\n'
```