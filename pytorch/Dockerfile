# base image Nvidia PyTorch image
FROM nvcr.io/nvidia/pytorch:19.01-py3

# install dependencies
RUN apt-get update
#RUN apt-get install sudo
RUN pip install --upgrade cython
RUN pip install \
  astroid==2.2.5 \
  cycler==0.10.0 \
  Cython>=0.29.13 \
  isort==4.3.21 \
  kiwisolver==1.1.0 \
  lazy-object-proxy==1.4.2 \
  matplotlib==3.1.1 \
  mccabe==0.6.1 \
  numpy==1.17.1 \
  Pillow==6.1.0 \
  protobuf==3.9.1 \
  pycocotools==2.0 \
  pylint==2.3.1 \
  pyparsing==2.4.2 \
  python-dateutil==2.8.0 \
  scipy==1.3.1 \
  six==1.12.0 \
  tensorboardX==1.8 \
  torch==1.2.0 \
  torchvision==0.4.0 \
  tqdm==4.35.0 \
  typed-ast==1.4.0 \
  wrapt>=1.10.11

# install python module for Bilateral filtering
RUN apt-get install swig -y

