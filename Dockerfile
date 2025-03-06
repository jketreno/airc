FROM ubuntu:oracular AS pytorch-build

SHELL [ "/bin/bash", "-c" ]

# Instructions Dockerfied from:
#
# https://github.com/pytorch/pytorch
#
# and
#
# https://pytorch.org/docs/stable/notes/get_start_xpu.html
# https://www.intel.com/content/www/us/en/developer/articles/tool/pytorch-prerequisites-for-intel-gpu/2-6.html
# 
#
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    gpg \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}


# ipex only supports python 3.11, so use 3.11 instead of latest oracular (3.12)

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    ca-certificates \
    ccache \
    cmake \
    curl \
    git \
    gpg-agent \
    less \
    libbz2-dev \
    libffi-dev \
    libjpeg-dev \
    libpng-dev \
    libreadline-dev \
    libssl-dev \
    libsqlite3-dev \
    llvm \
    nano \
    wget \
    zlib1g-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

#    python3 \
#    python3-pip \
#    python3-venv \
#    python3-dev \

RUN /usr/sbin/update-ccache-symlinks
RUN mkdir /opt/ccache && ccache --set-config=cache_dir=/opt/ccache

# Build Python in /opt/..., install it locally, then remove the build environment 
# collapsed to a single docker layer.
WORKDIR /opt
ENV PYTHON_VERSION=3.11.9

RUN wget -q -O - https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz | tar -xz \
    && cd Python-${PYTHON_VERSION} \
    && ./configure --prefix=/opt/python --enable-optimizations \
    && make -j$(nproc) \
    && make install \
    && cd /opt \
    && rm -rf Python-${PYTHON_VERSION}

WORKDIR /opt/pytorch

FROM ubuntu:oracular AS ze-monitor
# From https://github.com/jketreno/ze-monitor
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    debhelper \
    devscripts \
    cmake \
    git \
    libfmt-dev \
    libncurses-dev \
    rpm \
    rpm2cpio \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

RUN apt-get install -y \
    software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && apt-get install -y \
    libze-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

RUN git clone --depth 1 --branch v0.3.0-1 https://github.com/jketreno/ze-monitor /opt/ze-monitor
WORKDIR /opt/ze-monitor/build
RUN cmake .. \
    && make \
    && cpack

FROM pytorch-build AS pytorch

COPY --from=pytorch-build /opt/pytorch /opt/pytorch

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common \
    && add-apt-repository -y ppa:kobuk-team/intel-graphics \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libze-intel-gpu1 \
    libze1 \
    intel-ocloc \
    intel-opencl-icd \
    xpu-smi \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

RUN update-alternatives --install /usr/bin/python3 python3 /opt/python/bin/python3.11 2

# When cache is enabled SYCL runtime will try to cache and reuse JIT-compiled binaries.
ENV SYCL_CACHE_PERSISTENT=1

WORKDIR /opt/pytorch

RUN { \
    echo '#!/bin/bash' ; \
    update-alternatives --set python3 /opt/python/bin/python3.11 ; \
    echo 'source /opt/pytorch/venv/bin/activate' ; \
    echo 'bash -c "${@}"' ; \
    } > /opt/pytorch/shell ; \
    chmod +x /opt/pytorch/shell

RUN python3 -m venv --system-site-packages /opt/pytorch/venv

SHELL [ "/opt/pytorch/shell" ]

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
RUN pip3 freeze > /opt/pytorch/requirements.txt

SHELL [ "/bin/bash", "-c" ]

RUN { \
    echo '#!/bin/bash' ; \
    echo 'echo "Container: pytorch"' ; \
    echo 'set -e' ; \
    echo 'echo "Setting pip environment to /opt/pytorch"' ; \
    echo 'source /opt/pytorch/venv/bin/activate'; \
    echo 'if [[ "${1}" == "" ]] || [[ "${1}" == "shell" ]]; then' ; \
    echo '  echo "Dropping to shell"' ; \
    echo '  /bin/bash -c "source /opt/pytorch/venv/bin/activate ; /bin/bash"' ; \
    echo 'else' ; \
    echo '  exec "${@}"' ; \
    echo 'fi' ; \
    } > /entrypoint.sh \
    && chmod +x /entrypoint.sh

ENTRYPOINT [ "/entrypoint.sh" ]

FROM pytorch AS ipex-2.6.10

WORKDIR /opt
RUN git clone --branch release/xpu/2.6.10 --depth 1 https://github.com/intel/intel-extension-for-pytorch.git ipex-2.6.10
WORKDIR /opt/ipex-2.6.10

RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    intel-deep-learning-essentials-2025.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

# Requirements for building ipex / oneAPI...
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libspdlog-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

RUN python3 -m venv --system-site-packages /opt/ipex-2.6.10/venv

RUN { \
    echo '#!/bin/bash' ; \
    update-alternatives --set python3 /opt/python/bin/python3.11 ; \
    echo 'source /opt/intel/oneapi/setvars.sh' ; \
    echo 'source /opt/ipex-2.6.10/venv/bin/activate' ; \
    echo 'bash -c "${@}"' ; \
    } > /opt/ipex-2.6.10/shell ; \
    chmod +x /opt/ipex-2.6.10/shell

SHELL [ "/opt/ipex-2.6.10/shell" ]

#RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
RUN pip3 install -r requirements.txt

RUN git submodule update --init --recursive --depth 1

# Building ipex-2.6.10 wheel requires level-zero loader (libze-dev)
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libze-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

# torch needs to be installed
RUN pip3 install torch --index-url https://download.pytorch.org/whl/test/xpu

RUN python setup.py bdist_wheel

FROM pytorch AS ipex-llm-src

# Build ipex-llm from source

RUN git clone --depth 1 https://github.com/intel/ipex-llm.git /opt/ipex-llm

WORKDIR /opt/ipex-llm

RUN python3 -m venv --system-site-packages /opt/ipex-llm/venv
RUN { \
    echo '#!/bin/bash' ; \
    update-alternatives --set python3 /opt/python/bin/python3.11 ; \
    echo 'source /opt/ipex-llm/venv/bin/activate' ; \
    echo 'bash -c "${@}"' ; \
    } > /opt/ipex-llm/shell ; \
    chmod +x /opt/ipex-llm/shell

SHELL [ "/opt/ipex-llm/shell" ]

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
COPY --from=ipex-2.6.10 /opt/ipex-2.6.10/dist/intel_extension_for_pytorch-2.6.10*.whl /opt/wheels/
RUN for pkg in /opt/wheels/intel_extension_for_pytorch-2.6.10*.whl; do pip install $pkg[xpu-2-6]; done

WORKDIR /opt/ipex-llm/python/llm
RUN pip install requests wheel
RUN python setup.py clean --all bdist_wheel --linux

FROM airc AS jupyter

SHELL [ "/opt/airc/shell" ]

# BEGIN setup Jupyter
RUN pip install jupyter \
    jupyterlab==4.3.0a0 \
    jupyterhub==5.0.0 \
    notebook==7.3.0a0 \
    "jupyter-server-proxy>=4.1.2"
# END setup Jupyter

SHELL [ "/bin/bash", "-c" ]

RUN { \
    echo '#!/bin/bash' ; \
    echo 'echo "Container: airc jupyter"' ; \
    echo 'if [[ ! -e "/root/.cache/hub/token" ]]; then' ; \
    echo '  if [[ "${HF_ACCESS_TOKEN}" == "" ]]; then' ; \
    echo '    echo "Set your HF access token in .env as: HF_ACCESS_TOKEN=<token>" >&2' ; \
    echo '    exit 1' ; \
    echo '  else' ; \
    echo '    if [[ ! -d '/root/.cache/hub' ]]; then mkdir -p /root/.cache/hub; fi' ; \
    echo '    echo "${HF_ACCESS_TOKEN}" > /root/.cache/hub/token' ; \
    echo '  fi' ; \
    echo 'fi' ; \
    echo 'update-alternatives --set python3 /opt/python/bin/python3.11' ; \
    echo 'source /opt/intel/oneapi/setvars.sh' ; \
    echo 'source /opt/airc/venv/bin/activate' ; \
    echo 'if [[ "${1}" == "shell" ]]; then echo "Dropping to shell"; /bin/bash; exit $?; fi' ; \
    echo 'while true; do' ; \
    echo '  echo "Launching jupyter notebook"' ; \
    echo '  jupyter notebook \' ; \
    echo '    --notebook-dir=/opt/jupyter \' ; \
    echo '    --port 8888 \' ; \
    echo '    --ip 0.0.0.0 \' ; \
    echo '    --no-browser \' ; \
    echo '    --allow-root \' ; \
    echo '    --ServerApp.token= \' ; \
    echo '    --ServerApp.password= \' ; \
    echo '    --ServerApp.allow_origin=* \' ; \
    echo '    --ServerApp.base_url="/jupyter" \' ; \
    echo '    "${@}" \' ; \
    echo '    >> "/root/.cache/jupyter.log" 2>&1' ; \
    echo '  echo "jupyter notebook died ($?). Restarting."' ; \
    echo '  sleep 5' ; \
    echo 'done' ; \
    } > /entrypoint-jupyter.sh \
    && chmod +x /entrypoint-jupyter.sh

ENTRYPOINT [ "/entrypoint-jupyter.sh" ]

FROM pytorch AS airc

RUN python3 -m venv --system-site-packages /opt/airc/venv

# Don't install the full oneapi essentials; just the ones that we seem to need
RUN wget -O- https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB \
    | gpg --dearmor -o /usr/share/keyrings/oneapi-archive-keyring.gpg \
    && echo "deb [signed-by=/usr/share/keyrings/oneapi-archive-keyring.gpg] https://apt.repos.intel.com/oneapi all main" \
    | tee /etc/apt/sources.list.d/oneAPI.list \
    && apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    intel-oneapi-mkl-sycl-2025.0 \
    intel-oneapi-dnnl-2025.0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/{apt,dpkg,cache,log}

RUN { \
    echo '#!/bin/bash' ; \
    echo 'update-alternatives --set python3 /opt/python/bin/python3.11' ; \
    echo 'source /opt/intel/oneapi/setvars.sh' ; \
    echo 'source /opt/airc/venv/bin/activate' ; \
    echo 'if [[ "$1" == "" ]]; then bash -c; else bash -c "${@}"; fi' ; \
    } > /opt/airc/shell ; \
    chmod +x /opt/airc/shell

SHELL [ "/opt/airc/shell" ]

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/test/xpu
# Install ipex built in ipex-2.6.10
COPY --from=ipex-2.6.10 /opt/ipex-2.6.10/dist/*.whl /opt/wheels/
RUN for pkg in /opt/wheels/intel_extension_for_pytorch-2.6.10*.whl; do pip install $pkg[xpu-2-6]; done
# Install ipex-llm built in ipex-llm-src
COPY --from=ipex-llm-src /opt/ipex-llm/python/llm/dist/*.whl /opt/wheels/
RUN for pkg in /opt/wheels/ipex_llm*.whl; do pip install $pkg; done

COPY src/ /opt/airc/src/

# pydle does not work with newer asyncio due to coroutine
# being deprecated. Patch to work.
RUN pip3 install pydle transformers sentencepiece accelerate \
    && patch -d /opt/airc/venv/lib/python3*/site-packages/pydle \
    -p1 < /opt/airc/src/pydle.patch

# mistral fails with cache_position errors with transformers>4.40 (or at least it fails with the latest)
RUN pip install transformers==4.40

RUN pip3 install pydle transformers sentencepiece accelerate

# To get xe_linear and other Xe methods    
RUN pip3 install 'bigdl-core-xe-all>=2.6.0b'

# trl.core doesn't have what is needed with the default 'pip install trl' version
RUN pip install git+https://github.com/huggingface/trl.git@7630f877f91c556d9e5a3baa4b6e2894d90ff84c

# Needed by src/model-server.py
RUN pip install flask

SHELL [ "/bin/bash", "-c" ]

RUN { \
    echo '#!/bin/bash' ; \
    echo 'set -e' ; \
    echo 'if [[ ! -e "/root/.cache/hub/token" ]]; then' ; \
    echo '  if [[ "${HF_ACCESS_TOKEN}" == "" ]]; then' ; \
    echo '    echo "Set your HF access token in .env as: HF_ACCESS_TOKEN=<token>" >&2' ; \
    echo '    exit 1' ; \
    echo '  else' ; \
    echo '    if [[ ! -d '/root/.cache/hub' ]]; then mkdir -p /root/.cache/hub; fi' ; \
    echo '    echo "${HF_ACCESS_TOKEN}" > /root/.cache/hub/token' ; \
    echo '  fi' ; \
    echo 'fi' ; \
    echo 'echo "Container: airc"' ; \
    echo 'echo "Setting pip environment to /opt/airc"' ; \
    echo 'source /opt/intel/oneapi/setvars.sh'; \
    echo 'source /opt/airc/venv/bin/activate'; \
    echo 'if [[ "${1}" == "shell" ]] || [[ "${1}" == "/bin/bash" ]]; then' ; \
    echo '  echo "Dropping to shell"' ; \
    echo '  /bin/bash -c "source /opt/airc/venv/bin/activate ; /bin/bash"' ; \
    echo '  exit $?' ; \
    echo 'else' ; \
    echo '  while true; do' ; \
    echo '    echo "Launching model-server"' ; \
    echo '    python src/model-server.py \' ; \
    echo '      2>&1 | tee -a "/root/.cache/model-server.log"'; \
    echo '    echo "model-server died ($?). Restarting."' ; \
    echo '    sleep 5' ; \
    echo '  done &' ; \
    echo '  while true; do' ; \
    echo '    echo "Launching airc"' ; \
    echo '    python src/airc.py "${@}" \' ; \
    echo '      2>&1 | tee -a "/root/.cache/airc.log"' ; \
    echo '    echo "airc died ($?). Restarting."' ; \
    echo '    sleep 5' ; \
    echo '  done' ; \
    echo 'fi' ; \
    } > /entrypoint-airc.sh \
    && chmod +x /entrypoint-airc.sh

COPY --from=ze-monitor /opt/ze-monitor/build/ze-monitor-*deb /opt/
RUN dpkg -i /opt/ze-monitor-*deb

WORKDIR /opt/airc

ENTRYPOINT [ "/entrypoint-airc.sh" ]
