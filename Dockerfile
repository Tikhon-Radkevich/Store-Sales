FROM python:3.10

WORKDIR /app

COPY deployment/requirements.txt .

RUN pip install -r requirements.txt

COPY deployment .

CMD ["python3", "deployment/storesales/model_api.py"]
