FROM python:3.11

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

EXPOSE 8000
CMD ["uvicorn", "diabetes_endpoint:app", "--host", "0.0.0.0", "--port","8000"]
