# Use a PyTorch base image with CUDA and Jupyter support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the repository
COPY . .

# Expose the Jupyter Notebook port
EXPOSE 8888

# Start the Jupyter server
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--notebook-dir=/app/notebooks"]
