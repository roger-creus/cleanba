FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04
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

ENV PIP_DEFAULT_TIMEOUT=100 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_VERSION=1.3.2

# install python dependencies
COPY pyproject.toml pyproject.toml
COPY poetry.lock poetry.lock
COPY requirements.txt requirements.txt

# install requirements
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry config virtualenvs.create false
RUN poetry install --only main --no-root
RUN poetry run pip install --upgrade "jax[cuda12_pip]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN poetry run pip install -r requirements.txt

# Copy entrypoint script
COPY entrypoint.sh /usr/local/bin/
RUN chmod 777 /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]

# Make useful directories
RUN mkdir /src