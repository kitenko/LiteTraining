FROM ubuntu:22.04

RUN apt-get update && apt-get install -y wget

RUN sh -c "$(wget -O- https://github.com/deluan/zsh-in-docker/releases/download/v1.2.1/zsh-in-docker.sh)" -- \
    -t gnzh \
    -p git -p ssh-agent -p 'history-substring-search' \
    -a 'bindkey "\$terminfo[kcuu1]" history-substring-search-up' \
    -a 'bindkey "\$terminfo[kcud1]" history-substring-search-down' \
    -p https://github.com/zsh-users/zsh-autosuggestions \
    -p https://github.com/zsh-users/zsh-completions \
    -p https://github.com/esc/conda-zsh-completion \
    -p https://github.com/romkatv/zsh-defer.git \
    -p https://github.com/zdharma-continuum/fast-syntax-highlighting \

# Set the working directory
WORKDIR /app

COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl

# Install Pixi (specific version)
RUN curl -fsSL https://pixi.sh/install.sh | bash -s -- --version 0.41.3

# Make Pixi available in $PATH
ENV PATH="/root/.pixi/bin:$PATH"

# Defining an argument for selecting an environment with the default value "default"
ARG ENVIRONMENT=default

# We use the passed argument to set the environment via pixi
RUN pixi install -e $ENVIRONMENT

