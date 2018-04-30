FROM conda/miniconda3:latest

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY gunicorn_config.py .

COPY source /app

EXPOSE 5000

CMD ["gunicorn", "--chdir", "app", "--config", "./gunicorn_config.py", "app:app"]
