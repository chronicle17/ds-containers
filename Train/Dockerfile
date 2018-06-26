# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Set environment variables
ENV ds_blob_account # blob account name
ENV ds_blob_key # blob key
ENV ds_container # container name
ENV ds_model_filename # name of model filename to save

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Run app.py when the container launches
CMD ["python", "train_model.py"]