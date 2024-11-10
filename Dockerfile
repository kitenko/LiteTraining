FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update && apt-get install -y libgl1

RUN pip install --upgrade pip

COPY scripts/install_dependencies.sh /scripts/install_dependencies.sh

RUN bash /scripts/install_dependencies.sh
WORKDIR /app
