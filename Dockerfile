#FROM tensorflow/tensorflow:2.8.0-gpu
#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
FROM nvcr.io/nvidia/tensorflow:22.01-tf2-py3

ENV DEBIAN_FRONTEND noninteractive
ADD . /code
WORKDIR /code

RUN apt-get update -y
#RUN apt-get upgrade -y
RUN apt-get install -y protobuf-compiler
RUN protoc deeplab2/*.proto --python_out=.
#
#RUN apt-get install -y python3.8
#RUN apt-get install -y python3.8-dev
#RUN apt install -y python3-pip
RUN apt-get update -y
RUN python3.8 -m pip install --upgrade pip

#RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

#RUN apt install -y python3-docopt
#RUN apt install -y python3-clint
#RUN apt install -y python3-crontab
#RUN apt install -y python3-tablib
RUN python3.8 -m pip install -U Pillow 
RUN python3.8 -m pip install Cython 


RUN python3.8 -m pip install -r requirements.txt
RUN python3.8 -m pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

#WORKDIR /code/cocoapi/PythonAPI
#RUN make

WORKDIR /code


ENV PYTHONPATH=$PYTHONPATH:/code/
ENV PYTHONPATH=$PYTHONPATH:/code/models
#ENV PYTHONPATH=$PYTHONPATH:/code/cocoapi/PythonAPI
#ENV PATH=/usr/local/cuda-11.2/bin:$PATH
#ENV PATH=/usr/local/cuda/bin:$PATH
#ENV LD_LIBRARY_PATH /usr/local/cuda/lib:/usr/local/cuda-11.2/lib
#ENV PATH=$PATH:/usr/local/cuda/lib
#ENV PATH=$PATH:/usr/local/cuda-11.2/lib

#CMD python3.9 '/code/deeplab2/example.py'
#deeplab2/compile.sh gpu &&
CMD  python3.8 '/code/deeplab2/trainer/train.py' --config_file '/code/deeplab2/configs/emid/day_avi.textproto' --mode train_and_eval --model_dir '/code/exp/'
