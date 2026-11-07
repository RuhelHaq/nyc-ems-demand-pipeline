# Start from a small official Python image, matching our version
FROM python:3.14-slim

# Work inside /app in the container
WORKDIR /app

# Install packages first (Docker caches this step, so rebuilds are faster)
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the code, settings, data, and model
COPY config.yaml .
COPY src/ src/
COPY data/raw/ data/raw/
COPY models/champion/ models/champion/

# Tell the API to load the model from the folder
ENV MODEL_PATH=models/champion

# Start the API, reachable from outside the container
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]