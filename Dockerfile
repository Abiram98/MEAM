FROM pytorch/pytorch

RUN pip3 install pandas sklearn torchinfo

WORKDIR /code