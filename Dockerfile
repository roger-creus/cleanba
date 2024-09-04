FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive 

# install ubuntu dependencies
RUN apt-get -y update  && \
    apt-get -y install python3-pip xvfb ffmpeg git build-essential \
    && apt-get install -y software-properties-common \
    && apt-get -y update \
    && add-apt-repository universe
RUN apt-get -y update
RUN apt-get -y install python3
RUN apt-get -y install python3-pip
RUN ln -s /usr/bin/python3 /usr/bin/python

# install python dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# install requirements
RUN pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# Optionally set the PYTHONPATH
ENV PYTHONPATH="/src:${PYTHONPATH}"

# Make useful directories
RUN mkdir /dataset
RUN mkdir /tmp_log
RUN mkdir /final_log
RUN mkdir /src