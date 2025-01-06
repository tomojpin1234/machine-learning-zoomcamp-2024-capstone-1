IMAGE_TAG=post_engagement:latest
DOCKER_FLAGS=-t $(IMAGE_TAG)

# Target to install test dependencies and and run python jupyter
start_jupyter:
	poetry install --with test && poetry run python -m ipykernel install --user --name=machine-learning-zoomcamp-2024-capstone-1 --display-name "ML-Zoomcamp-2024-Capstone-1-Kernel-Poetry"

# Target to install deploy dependencies and run server
server: check-models
	poetry install --with deploy && poetry run python3 predict.py

# Target to build Docker container with the latest tag
container:
	docker build $(DOCKER_FLAGS) .

# Check if the model files exist
check-models:
	@echo "Checking if model files exist..."
	@if [ ! -f "models/model_lgbmr.bin" ]; then \
		echo "Error: models/model_lgbmr.bin not found."; \
		echo "Please run 'make train_full' first."; \
		exit 1; \
	fi

# Stop previous container
stop_container:
	# Stop any existing container running on port 8080
	@echo "Checking if a previous container is running..."
	@docker ps -q --filter "ancestor=post_engagement:latest" | xargs -r docker stop
	@docker ps -aq --filter "ancestor=post_engagement:latest" | xargs -r docker rm

# Target to build and run the Docker container
run_container: check-models container stop_container
	# Run the new container
	@echo "Starting the new container..."
	@docker run -d -p 8080:8080 $(DOCKER_FLAGS)
	@echo "Container started successfully."
	@echo "Open your browser and go to http://localhost:8080 to access the application."


# Target to install test dependencies and run training script
train:
	poetry install --with test --no-root && poetry run python3 train.py