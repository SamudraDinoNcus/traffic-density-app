FROM python:3.10-slim

WORKDIR /app

# install system deps
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# copy project
COPY . .

# install python deps
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# streamlit config
ENV PORT=7860

EXPOSE 7860

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]