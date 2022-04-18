FROM ubuntu

RUN apt update -y
RUN apt install git python pip -y
RUN pip3 install numpy pandas sklearn
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN apt-get install git-lfs

WORKDIR /home/
RUN mkdir benchmark
RUN mkdir .git

COPY dataloader.py ./benchmark/
COPY benchmark/lsh_benchmark.py benchmark/
COPY src/lsh.py ./benchmark
COPY .git/ .git/

WORKDIR benchmark/

CMD python3 lsh_benchmark.py
