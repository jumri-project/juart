# ----- 1. First stage to build ismrmrd and siemens_to_ismrmrd -----

# see here for information on how to install siemens_to_ismrmrd in a containaer
# https://github.com/kspaceKelvin/python-ismrmrd-server/blob/master/docker/Dockerfile

FROM ubuntu:jammy AS mrd_converter
ARG  DEBIAN_FRONTEND=noninteractive
ENV  TZ=Europe/Berlin

RUN  apt-get update && apt-get install -y git cmake g++ libhdf5-dev libxml2-dev libxslt1-dev libboost-all-dev libfftw3-dev libpugixml-dev
RUN  mkdir -p /opt/code

# ISMRMRD library
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/ismrmrd.git && \
    cd ismrmrd && \
    git checkout d364e03 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
    make -j $(nproc) && \
    make install

# siemens_to_ismrmrd converter
RUN cd /opt/code && \
    git clone https://github.com/ismrmrd/siemens_to_ismrmrd.git && \
    cd siemens_to_ismrmrd && \
    git checkout v1.2.11 && \
    mkdir build && \
    cd build && \
    cmake ../ && \
     make -j $(nproc) && \
    make install

# Create archive of ISMRMRD libraries (including symlinks) for second stage
RUN cd /usr/local/lib && tar -czvf libismrmrd.tar.gz libismrmrd*

# ----- 2. Create a devcontainer without all of the build dependencies of MRD -----

# FROM quay.io/condaforge/miniforge3
# The quay image is based on ubuntu focal, which is too old to run ismrmrd because
# of incompatible libc versions.

# https://github.com/conda-forge/miniforge-images/blob/master/ubuntu/Dockerfile
FROM ubuntu:jammy

ARG MINIFORGE_NAME=Miniforge3
ARG MINIFORGE_VERSION=24.11.3-2
ARG TARGETPLATFORM

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}

# 1. Install just enough for conda to work
# 2. Keep $HOME clean (no .wget-hsts file), since HSTS isn't useful in this context
# 3. Install miniforge from GitHub releases
# 4. Apply some cleanup tips from https://jcrist.github.io/conda-docker-tips.html
#    Particularly, we remove pyc and a files. The default install has no js, we can skip that
# 5. Activate base by default when running as any *non-root* user as well
#    Good security practice requires running most workloads as non-root
#    This makes sure any non-root users created also have base activated
#    for their interactive shells.
# 6. Activate base by default when running as root as well
#    The root user is already created, so won't pick up changes to /etc/skel
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        git \
        tini \
        > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean --tarballs --index-cache --packages --yes && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean --force-pkgs-dirs --all --yes  && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc && \
    if [ ${MINIFORGE_NAME} = "Mambaforge"* ]; then \
        echo '. ${CONDA_DIR}/etc/.mambaforge_deprecation.sh' >> /etc/skel/.bashrc && \
        echo '. ${CONDA_DIR}/etc/.mambaforge_deprecation.sh' >> ~/.bashrc; \
    fi

COPY mambaforge_deprecation.sh ${CONDA_DIR}/etc/.mambaforge_deprecation.sh

ENTRYPOINT ["tini", "--"]
CMD [ "/bin/bash" ]

ENV DEBIAN_FRONTEND=noninteractive

# Copy ISMRMRD files from last stage
COPY --from=mrd_converter /usr/local/include/ismrmrd        /usr/local/include/ismrmrd/
COPY --from=mrd_converter /usr/local/share/ismrmrd          /usr/local/share/ismrmrd/
COPY --from=mrd_converter /usr/local/bin/ismrmrd*           /usr/local/bin/
COPY --from=mrd_converter /usr/local/lib/libismrmrd.tar.gz  /usr/local/lib/
RUN cd /usr/local/lib && tar -zxvf libismrmrd.tar.gz && rm libismrmrd.tar.gz && ldconfig

# Copy siemens_to_ismrmrd from last stage
COPY --from=mrd_converter /usr/local/bin/siemens_to_ismrmrd  /usr/local/bin/siemens_to_ismrmrd

RUN apt-get update \
    # (TODO) A recommendation that comes along with 
    # build-essential is required to pip install finufft
    # && apt-get install -y --no-install-recommends \
    && apt-get install -y \
    build-essential \
    ssh \
    g++ \
    cmake \
    git \
    libboost-all-dev \
    xsdcxx \
    libxerces-c-dev \
    libhdf5-serial-dev \
    h5utils \
    hdf5-tools \
    libtinyxml-dev \
    libxml2-dev \
    libxslt1-dev \
    libpugixml-dev \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/

RUN mamba env update \
    --name "base" \
    --file /tmp/environment.yml \
    && \
    mamba clean --all -f -y
