FROM ubuntu

RUN mkdir data
WORKDIR /home

RUN apt update -y
RUN apt install git python pip -y
RUN pip3 install numpy

CMD python3 lsh.py search /data/test_data.npy /data/store.db
