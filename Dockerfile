# Use an official Python runtime as a parent image (we'll use 3.9 as discussed)
FROM python:3.12.10-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the requirements file first to leverage Docker's layer caching
COPY requirements.txt ./

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your project's files into the container
# This includes your src, models, and data folders
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# The command to run your app when the container launches
# The path must match your project structure
CMD ["streamlit", "run", "src/app/app.py"]