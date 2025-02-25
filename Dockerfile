# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file to install dependencies
COPY requirements.txt .

# Install system dependencies and Python packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

# Copy the app code and model file into the container
COPY app.py .
COPY model/gender_model_vgg16.h5 ./model/

# Expose the port Gradio will run on
EXPOSE 7860

# Command to run the Gradio app
CMD ["python", "app.py"]