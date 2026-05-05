FROM python:3.11-slim

WORKDIR /app

COPY common/ ./common/

COPY Master/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY Master/ .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7000"]