# 1. Use the same lightweight Python base engine
FROM python:3.11-slim

# 2. Set the default working folder inside the container
WORKDIR /app

# 3. Copy our required libraries (we will create a small list file for it next)
COPY requirements.txt .

# 4. Install Streamlit and network clients inside the container
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy our frontend app.py layout file into the container
COPY . .

# 6. Open the communication gate for Streamlit web interface traffic
EXPOSE 8501

# 7. Boot up the Streamlit interface automatically when this container turns on
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
