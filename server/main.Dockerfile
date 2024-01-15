FROM python:3.10-slim as builder

WORKDIR /app

RUN apt-get update \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip \
    && pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim as release

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

COPY --from=builder /usr/local/lib/python3.10/site-packages/ /usr/local/lib/python3.10/site-packages/
COPY --from=builder /usr/local/bin/gunicorn /usr/local/bin

COPY --from=builder /app .

VOLUME /app/log

EXPOSE 5555

CMD gunicorn --bind 0.0.0.0:5555 app:app