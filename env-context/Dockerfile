FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu22.04:latest

ENV DEBIAN_FRONTEND=noninteractive

# Copy your environment files into the build context
COPY environment-dev.yml requirements-dev.txt ./

# Create the conda environment
RUN conda env create -f environment-dev.yml && \
    conda clean -ay

# Ensure the environment is activated
ENV CONDA_DEFAULT_ENV=xray-training
ENV PATH /opt/miniconda/envs/xray-training/bin:$PATH