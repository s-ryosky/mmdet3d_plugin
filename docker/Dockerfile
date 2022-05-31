ARG PYTORCH="1.9.0"
ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0 7.2 7.5 8.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Update the CUDA Linux GPG repository key
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install linux packages
RUN apt-get update && \
    apt-get install -y \
        ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
        libgl1 libgomp1 mesa-utils x11-apps sudo locales && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install MMCV, MMDetection and MMSegmentation
RUN pip install seaborn==0.11.2
ARG PYTORCH
ARG CUDA
ARG MMCV="1.5.0"
ARG MMDET="2.19.0"
ARG MMSEG="0.20.0"
RUN ["/bin/bash", "-c", "pip install --no-cache-dir mmcv-full==${MMCV} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA//./}/torch${PYTORCH}/index.html"]
RUN pip install --no-cache-dir mmdet==${MMDET} mmsegmentation==${MMSEG}

# Install MMDetection3D
RUN conda clean --all
COPY . /mmdetection3d
WORKDIR /mmdetection3d
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN pip install -r requirements/optional.txt
RUN pip install --ignore-installed protobuf==3.8.0

# Set time zone
ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set the locale
RUN locale-gen ja_JP.UTF-8
ENV LANG ja_JP.UTF-8
ENV LANGUAGE ja:en
ENV LC_ALL ja_JP.UTF-8

# TensorBoard
EXPOSE 6006

# IPython
EXPOSE 8888

# User env
ARG USERNAME=docker
ARG UID=1000
ARG GID=1000
RUN useradd -m $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    usermod --shell /bin/bash $USERNAME && \
    usermod -aG sudo $USERNAME && \
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers.d/$USERNAME && \
    chmod 0440 /etc/sudoers.d/$USERNAME && \
    usermod --uid $UID $USERNAME && \
    groupmod --gid $GID $USERNAME
