# Dockerfile, Image and Container.
FROM python:3.11.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install the dependencies
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port your application will run on
EXPOSE 3000

# Run the application
CMD ["python", "app.py"]
