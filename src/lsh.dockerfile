# Use '..' as build context

FROM ubuntu

RUN mkdir data
WORKDIR /home

RUN apt update -y
RUN apt install git python pip -y

COPY data/ /data/
COPY src/lsh.py ./

ENTRYPOINT python3 lsh.py
