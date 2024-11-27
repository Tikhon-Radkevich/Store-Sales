FROM python:3.10

WORKDIR /storesales

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

CMD ["python3", "deployment/test_script.py"]
