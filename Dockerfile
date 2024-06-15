FROM python:3.10.3-slim-buster

WORKDIR /workspace

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

ARG MODEL_URL

ENV MODEL_URL=${MODEL_URL}

ENV PYTHONUNBUFFERED=1

ENV HOST 0.0.0.0

RUN apt-get update && apt-get install -y wget unzip && \
    wget -O model.h5 "${MODEL_URL}"

COPY . .

EXPOSE 8080

CMD ["python", "main.py"]
