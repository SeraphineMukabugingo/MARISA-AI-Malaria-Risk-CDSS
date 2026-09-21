FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir streamlit==1.36.0 plotly==5.22.0 requests==2.32.3 streamlit-folium==0.22.0 folium==0.16.0

COPY streamlit_app/app.py .

EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
