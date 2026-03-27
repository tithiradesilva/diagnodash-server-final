# 1. Use a lightweight Python base image
FROM python:3.9-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Force logs to show up immediately in the Render console
ENV PYTHONUNBUFFERED=1

# 4. Copy dependency file and install Python libraries (CPU-only)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy all your V2 code files and the model.pth into the container
COPY . .

# 6. Tell the cloud which port we are using
EXPOSE 5000

# 7. Start the server using Gunicorn (Production Grade)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "2", "--timeout", "120", "server:app"]