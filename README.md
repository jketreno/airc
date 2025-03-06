# AIRC (pronounced Eric)

AI is Really Cool

NOTE: If running on an Intel Arc A series graphics processor, fp64 is not supported and may need to either be emulated or have the model quantized.

This project provides container definitions that will provide PyTorch 2.6 with
Intel's LLM project. In addition, it provides a small local chat server and an IRC client to provide a chat bot.

# Installation

This project uses docker containers to build. As this was originally
written to work on an Intel Arc B580 (Battlemage), it requires a
kernel that supports that hardware, such as the one documented
at [Intel Graphics Preview](https://github.com/canonical/intel-graphics-preview), which runs in Ubuntu Oracular (24.10)..

NOTE: You need 'docker compose' installed. See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

## Want to run under WSL2? No can do...

https://www.intel.com/content/www/us/en/support/articles/000093216/graphics/processor-graphics.html

The A- and B-series discrete GPUs do not support SR-IOV, required for
the GPU partitioning that Microsoft Windows uses in order to support GPU acceleration in WSL.

## Building

NOTE: You need 'docker compose' installed. See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)


```bash
git clone https://github.com/jketreno/airc
cd airc
docker compose build
```

## Running

In order to download the models, you need to have a Hugging Face
token. See https://huggingface.co/settings/tokens for information
on obtaining a token.

Edit .env to add the following:

```.env
HF_ACCESS_TOKEN=<access token from huggingface>
```

NOTE: Models downloaded by most examples will be placed in the
./cache directory, which is bind mounted to the container.

### AIRC

To launch the airc shell interactively, with the pytorch 2.6
environment loaded, use the default entrypoint to launch a shell:

```bash
docker compose run --rm airc shell
```

Once in the shell, you can then launch the model-server.py and then
the airc.py client:

```bash
docker compose run --rm airc shell
src/airc.py --ai-server=http://localhost:5000 &
src/model-server.py
```

By default, src/airc.py will connect to irc.libera.chat on the airc-test
channel. See `python src/airc.py --help` for options.

By separating the model-server into its own process, you can develop
and tweak the chat backend without losing the IRC connection established
by airc.

### Jupyter

```bash
docker compose up jupyter -d
```

The default port for inbound connections is 8888 (see docker-compose.yml).
$(pwd)/jupyter is bind mounted to /opt/juypter in the container, which is where notebooks will be saved by default.

To access the jupyter notebook, go to `https://localhost:8888/jupyter`.
