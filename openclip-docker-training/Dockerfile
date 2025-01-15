
FROM ubuntu:22.04


RUN apt-get update && \
    apt-get -y install --no-install-recommends \
    build-essential \
    ca-certificates \
    openjdk-8-jdk-headless \
    python3-dev \
    curl \
    vim \
    psmisc \
    git \
    zip \
    unzip \
    && rm -rf /var/lib/apt/lists/* \
    && curl -O https://bootstrap.pypa.io/get-pip.py \
    && python3 get-pip.py

RUN apt-get update
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" \
    && unzip awscliv2.zip \
    && ./aws/install


RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install open_clip_torch mlflow boto3 torchinfo psutil pynvml


RUN mkdir -p /home/clip-training/
COPY *.py /home/clip-training/


RUN mkdir -p /home/clip-training/data
RUN mkdir -p /home/clip-training/models

# Entrypoint script for the docker image
ENTRYPOINT ["python3", "/home/clip-training/entrypoint.py"]





