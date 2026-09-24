FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    dash==2.17.1 \
    plotly==5.22.0 \
    requests==2.32.3 \
    pandas==2.2.2

COPY dash_app/app.py .

EXPOSE 8050

CMD ["python", "app.py"]
