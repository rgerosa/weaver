# import from dockerhub of cuda base
FROM nvidia/cuda:11.4.0-runtime-centos7

# Maintainer label
LABEL maintainer="Raffaele Gerosa <rgerosa@ucsd.edu>"

# Copy the Miniconda3 file that needs to be periodically updated
COPY Miniconda3-latest-Linux-x86_64.sh .
RUN set -x chmod Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b

# Set conda environment 
ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

# Installa mamba
RUN conda install -y mamba -c conda-forge

# Create the environment for weaver:
COPY weaver-environment-no-torch.yaml .
RUN conda env create -f weaver-environment-no-torch.yaml

# Initialize conda bash config
RUN conda init bash

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "weaver", "/bin/bash", "-c"]

# Install torch by hand
RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
