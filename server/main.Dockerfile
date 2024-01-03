FROM python:3.8.18-slim-bullseye as server

WORKDIR /app

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6 -y

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

VOLUME /app/log

EXPOSE 5555

CMD [ "python3", "app.py"]
