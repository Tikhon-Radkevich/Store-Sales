FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

CMD ["python3", "deployment/test_script.py"]
