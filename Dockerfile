# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model file and application code
COPY ner_model/ /app/ner_model/
COPY app.py .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Define environment variables
ENV API_USERNAME=admin
ENV API_PASSWORD=password

# Run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]