FROM nvidia/cuda:10.1-cudnn7-devel

RUN apt-get update
RUN apt-get -y install python3.7-dev python3-pip libsm6 libxext6 libxrender1
RUN apt-get -y install -y --no-install-recommends libnvinfer6=6.0.1-1+cuda10.1 libnvinfer-dev=6.0.1-1+cuda10.1 libnvinfer-plugin6=6.0.1-1+cuda10.1
RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install "tensorflow==2.1.0" pandas ray[rllib] pybullet tensorflow-probability
