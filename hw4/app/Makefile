# Makefile

# Default target
all: start-services run-prep pull-model

# Set environment variable and start Docker services
start-services:
	export POSTGRES_HOST="localhost" && docker-compose up -d

# Run the prep.py script
run-prep:
	python prep.py

# Pull the phi3 model into ollama
pull-model:
	docker-compose exec -T ollama ollama pull phi3

# Stop all services
stop:
	docker-compose down

# Clean up (stop services and remove volumes)
clean:
	docker-compose down -v

.PHONY: all start-services run-prep pull-model stop clean