# syntax=docker/dockerfile:experimental
FROM python:3.7

WORKDIR /app

COPY requirements.txt /app
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip install -r requirements.txt
RUN --mount=type=cache,id=custom-pip,target=/root/.cache/pip pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html

COPY . /app
EXPOSE 5001

CMD python app.py
