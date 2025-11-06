# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY .streamlit/ ./.streamlit/

# Expose Streamlit default port
EXPOSE 8501

# Set environment variable for port (can be overridden)
ENV PORT=8501

# Run the Streamlit app
CMD streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
