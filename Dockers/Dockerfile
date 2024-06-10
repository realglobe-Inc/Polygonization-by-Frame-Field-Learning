FROM mambaorg/micromamba:jammy-cuda-12.1.1

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive

ARG CONTAINER_USER=user
ARG CONTAINER_USER_ID=57440
ARG CONTAINER_USER_GID=57440

# ソフトウェアインストール
# libgl1-mesa-dev : opencv用
USER root
RUN apt update --fix-missing && \
    apt upgrade --fix-missing -y && \
    apt install -y git libgl1-mesa-dev &&\
    apt autoremove && \
    rm -rf /var/lib/apt/lists/* && \
    apt clean

# ユーザーを指定したID（ホストのユーザーIDなど）に変更する
RUN usermod "--login=${CONTAINER_USER}" "--home=/home/${CONTAINER_USER}" \
        --move-home "-u ${CONTAINER_USER_ID}" "${MAMBA_USER}" && \
    groupmod "--new-name=${CONTAINER_USER}" \
        "-g ${CONTAINER_USER_GID}" "${MAMBA_USER}" && \
    # Update the expected value of MAMBA_USER for the
    # _entrypoint.sh consistency check.
    echo "${CONTAINER_USER}" > "/etc/arg_mamba_user"
ENV MAMBA_USER=$CONTAINER_USER

USER $MAMBA_USER
COPY --chown=$MAMBA_USER:$MAMBA_USER env.yaml /tmp/env.yaml
RUN micromamba install -y -n base -f /tmp/env.yaml && \
    micromamba clean --all --yes
