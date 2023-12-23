ARG BUILD_ARCH=amd64

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND noninteractive

RUN apt update -y \
    && apt install -y build-essential git bc software-properties-common libgomp1 \
    # && apt install -y msttcorefonts -qq \  # https://stackoverflow.com/questions/42097053/matplotlib-cannot-find-basic-fonts
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt install -y python3.9-dev python3-pip \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --set python3 /usr/bin/python3.9 \
    && python3 -m pip install --upgrade pip setuptools wheel

RUN git clone https://github.com/aucad/new-experiments.git "/usr/src/new-experiments"

RUN pip3 install -r "/usr/src/new-experiments/requirements.txt" --user

WORKDIR ./usr/src/new-experiments