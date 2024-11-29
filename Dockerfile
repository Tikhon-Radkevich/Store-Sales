FROM python:3.10

COPY deployment/requirements.txt .

RUN pip install -r requirements.txt

COPY deployment deployment
#COPY storesales storesales

EXPOSE 8000

CMD ["uvicorn", "deployment.sales_api.model_api:app", "--host", "0.0.0.0", "--port", "8000"]
