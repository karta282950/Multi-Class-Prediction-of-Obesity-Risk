FROM nvidia/cuda:11.0-base

WORKDIR /app

COPY . /app

RUN apt-get update && apt-get install -y python3-pip && \
    pip3 install --no-cache-dir -r requirements.txt

CMD ["python3", "tune.py"]