ARG CUDA_VER=12.8.1
ARG LINUX_VER=ubuntu24.04
FROM nvidia/cuda:${CUDA_VER}-base-${LINUX_VER}

ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VER=3.12.11
ENV PATH=/opt/conda/bin:$PATH
ENV PYTHON_VERSION=${PYTHON_VER}

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# Create a conda group and assign it as root's primary group
RUN <<EOF
groupadd conda
usermod -g conda root
EOF

# Ownership & permissions based on https://docs.anaconda.com/anaconda/install/multi-user/#multi-user-anaconda-installation-on-linux
COPY --from=condaforge/miniforge3:latest --chown=root:conda --chmod=770 /opt/conda /opt/conda

# Ensure new files are created with group write access & setgid. See https://unix.stackexchange.com/a/12845
RUN chmod g+ws /opt/conda

RUN <<EOF
# Ensure new files/dirs have group write permissions
umask 002
# install expected Python version
conda install -y -n base "python~=${PYTHON_VERSION}.0=*_cpython"
conda update --all -y -n base
find /opt/conda -follow -type f -name '*.a' -delete
find /opt/conda -follow -type f -name '*.pyc' -delete
conda clean -afy
EOF

# Reassign root's primary group to root
RUN usermod -g root root

# ensure conda environment is always activated
RUN <<EOF
ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
echo ". /opt/conda/etc/profile.d/conda.sh; conda activate base" >> /etc/skel/.bashrc
echo ". /opt/conda/etc/profile.d/conda.sh; conda activate base" >> ~/.bashrc
EOF

# update und upgrade
RUN <<EOF
apt-get update
apt-get upgrade -y
EOF


RUN <<EOF
# build-essential is required to pip install finufft
apt-get install -y --no-install-recommends 
apt-get install -y \
    build-essential \
    ssh \
    libfftw3-dev \
    pkg-config \
    git \
    libxerces-c-dev \
    libsundials-dev \
    libginac-dev \
    ginac-tools \
    libginac11 \
    libhdf5-dev \
    libismrmrd-dev \
    libboost-all-dev
apt-get autoremove -y 
apt-get clean 
EOF

# Install JEMRIS
RUN <<EOF 
git clone https://github.com/JEMRIS/jemris.git
cd jemris
mkdir build; cd build; cmake -DSKIP_CONDA=1 ../
make 
make install
cd ../..
rm -rf jemris
EOF

# Allow execution of mpi as ROOT user
ENV OMPI_ALLOW_RUN_AS_ROOT=1
ENV OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 

# Install bart
RUN apt-get install -y bart

COPY environment.yml /tmp/

RUN mamba env update \
    --name "base" \
    --file /tmp/environment.yml \
    && \
    mamba clean --all -f -y

# remove install lists
RUN rm -rf /var/lib/apt/lists/*
