# Using python 3.8 since that's what I developed it on,
# In the future it would be important to update for compatibility and security
FROM python:3.8

# Set a working directory
WORKDIR /app

# Copy this repository's files
COPY . ./

# Run updates, install and setup
RUN python3 -m pip install --upgrade pip && python3 -m pip install -r requirements.txt

# Expose the port where we'll serve our app
EXPOSE 8000

# Train model once, to ensure it is available. 
# In production this could be replaced with an environment variable that points to the model url, or some other means.
RUN python3 model/model.py

# Run our application programming interface
CMD python api/main.py



