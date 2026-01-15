FROM python:3.8-slim

WORKDIR /app

# Copy project files
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run tests
RUN pytest tests/ -v --cov=ev_qa_framework

# Default command
CMD ["python", "-m", "pytest", "tests/", "-v"]
