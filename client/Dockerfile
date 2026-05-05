FROM python:3.11-slim

WORKDIR /app

COPY common/ ./common/

COPY Client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Client/ .

CMD ["python", "load_generator.py"]