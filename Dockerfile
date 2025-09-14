## Use Ubuntu
FROM ubuntu:22.04


### ================== System Setups ======================
## Install System Dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt -y upgrade
RUN apt -y install \
    coreutils wget vim git \
    gcc-9 g++-9 \
    make cmake \
    libboost-dev libboost-program-options-dev \
    openmpi-bin openmpi-doc libopenmpi-dev \
    python3 python3-pip python3-venv \
    graphviz

RUN pip3 install --upgrade pip
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 90 --slave /usr/bin/g++ g++ /usr/bin/g++-9

### ================== Finalize ==========================
## Move to the application directory
WORKDIR /app
### ======================================================
