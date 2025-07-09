FROM python:3.9-slim

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y git ffmpeg libsm6 libxext6 \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

ENV BUCKET_NAME=product-catalogues
ENV INPUT_FILE=combined_products.json
ENV OUTPUT_FILE=combined_enriched.json

CMD ["python", "main.py"]