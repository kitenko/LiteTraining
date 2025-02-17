FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]

COPY scripts/install_dependencies.sh /scripts/install_dependencies.sh

RUN mkdir -p /data

RUN bash /scripts/install_dependencies.sh

WORKDIR /app
