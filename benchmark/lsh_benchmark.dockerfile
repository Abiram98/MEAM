FROM ubuntu

RUN apt update -y
RUN apt install git python pip -y
RUN pip3 install numpy pandas
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu

RUN mkdir data
WORKDIR /home/
COPY dataloader.py ./
COPY data/ /data/
COPY benchmark/lsh_benchmark.py ./
COPY src/lsh.py ./

#CMD python3 lsh.py search /data/test_data.npy /data/store.db
CMD python3 lsh_benchmark.py
