# Use an official Python image as base
FROM python:3.12

# Set the working directory in the container
WORKDIR /app

# Copy the project files into the container
COPY . .

# Install Dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose the port FastAPI runs on
EXPOSE 8080

# Run the FastAPI Application
CMD [ "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080" ]