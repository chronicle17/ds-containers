# Use an official Python runtime as a parent image
FROM python:3.6-slim

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV ds_blob_account # Azure Storage Account name
ENV ds_blob_key # Azure storage account key
ENV ds_container # Azure blob container name
ENV ds_model_filename # model file name

COPY falcon_gateway.py /falcon_gateway.py
COPY handle_data.py /handle_data.py

# Run gunicorn with falcon_gateway when the container launches
CMD ["/usr/local/bin/gunicorn", "-b 0.0.0.0:80", "falcon_gateway:app"]
#ENTRYPOINT ["/usr/local/bin/gunicorn", "-b 0.0.0.0:80", "falcon_gateway:app"]