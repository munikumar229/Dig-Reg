# Start from the official Python slim image
FROM python:3.12.3-slim
# Set the working directory in the container
WORKDIR /app
# Copy the current directory contents into the container at /app
COPY . .
# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
# Make port 8501 available to the world outside this container
EXPOSE 8501
# Run app.py when the container launches
CMD ["uvicorn", "src.main:app", "--host=0.0.0.0", "--port=8501"]

