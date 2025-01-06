FROM python:3.13-slim

WORKDIR /app

# Install Poetry
RUN pip install poetry

# Copy project files
COPY ["pyproject.toml", "poetry.lock", "./"]

# Install libgomp.so.1 in the Docker Container
RUN apt-get update && apt-get install -y libgomp1

# Install dependencies without virtual environment (makes them available system-wide)
RUN poetry config virtualenvs.create false && poetry install --with deploy

# Copy additional app files
COPY ["predict.py", "./"]

# Copy the templates directory and its contents
COPY templates/ templates/

# Copy the data directory and its contents
COPY data/ data/

# Copy the data directory and its contents
COPY models/ models/

# Expose the application port
EXPOSE 8080

# Start Gunicorn
ENTRYPOINT [ "gunicorn", "-b", "0.0.0.0:8080", "predict:app" ]
