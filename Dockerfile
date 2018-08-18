FROM ubuntu:16.04

# Install requirements
RUN apt-get update && apt-get install -y \
    build-essential \
    checkinstall \
    libreadline-gplv2-dev \
    libncursesw5-dev \
    libssl-dev \
    libsqlite3-dev \
    tk-dev \
    libgdbm-dev \
    libc6-dev \
    libbz2-dev \
    zlib1g-dev \
    openssl \
    libffi-dev \
    python3-dev \
    python3-setuptools \
    wget \
    cmake \
    swig

# Pull down Python 3.7, build, and install
WORKDIR /tmp/Python37
RUN wget https://www.python.org/ftp/python/3.7.0/Python-3.7.0.tar.xz && \
    tar xvf Python-3.7.0.tar.xz

WORKDIR  /tmp/Python37/Python-3.7.0
RUN ./configure && \
    make && \
    make install

# Update pip packages
RUN pip3 install -U pip setuptools wheel virtualenv

# Install requirements
COPY requirements.txt  /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

WORKDIR /deep-rl
CMD ["/bin/bash"]
