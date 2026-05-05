FROM python:3.11-slim

WORKDIR /app

COPY common/ ./common/

COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY client/ .

CMD ["python", "load_generator.py"]