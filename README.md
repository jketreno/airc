# AIRC (pronounced Eric)

AI is Really Cool

This project provides a simple IRC chat client. It runs the neuralchat model, enhanced with a little bit of RAG to fetch news RSS feeds.

Internally, it is built using PyTorch 2.6 and the Intel IPEX/LLM.

NOTE: If running on an Intel Arc A series graphics processor, fp64 is not supported and may need to either be emulated or have the model quantized. It has been a while since I've had an A series GPU to test on, so if you run into problems please file an [issue](https://github.com/jketreno/airc/issues)--I have some routines I can put in, but don't have a way to test them. 

# Installation

This project uses docker containers to build. As this was originally written to work on an Intel Arc B580 (Battlemage), it requires a kernel that supports that hardware, such as the one documented at [Intel Graphics Preview](https://github.com/canonical/intel-graphics-preview), which runs in Ubuntu Oracular (24.10)..

NOTE: You need 'docker compose' installed. See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)

## Want to run under WSL2? No can do...

https://www.intel.com/content/www/us/en/support/articles/000093216/graphics/processor-graphics.html

The A- and B-series discrete GPUs do not support SR-IOV, required for the GPU partitioning that Microsoft Windows uses in order to support GPU acceleration in WSL.

## Building

NOTE: You need 'docker compose' installed. See [Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)


```bash
git clone https://github.com/jketreno/airc
cd airc
docker compose build
```

## Running

In order to download the models, you need to have a Hugging Face token. See https://huggingface.co/settings/tokens for information on obtaining a token.

Edit .env to add the following:

```.env
HF_ACCESS_TOKEN=<access token from huggingface>
```

NOTE: Models downloaded by most examples will be placed in the ./cache directory, which is bind mounted to the container.

### AIRC

To launch the airc shell interactively, with the pytorch 2.6 environment loaded, use the default entrypoint to launch a shell:

```bash
docker compose run --rm airc shell
```

Once in the shell, you can then launch the model-server.py and then the airc.py client:

```bash
docker compose run --rm airc shell
src/airc.py --ai-server=http://localhost:5000 &
src/model-server.py
```

By default, src/airc.py will connect to irc.libera.chat on the airc-test channel. See `python src/airc.py --help` for options.

By separating the model-server into its own process, you can develop and tweak the chat backend without losing the IRC connection established by airc.

### Jupyter

```bash
docker compose up jupyter -d
```

The default port for inbound connections is 8888 (see docker-compose.yml). $(pwd)/jupyter is bind mounted to /opt/juypter in the container, which is where notebooks will be saved by default.

To access the jupyter notebook, go to `https://localhost:8888/jupyter`.

### Monitoring

You can run `ze-monitor` within the launched containers to monitor GPU usage.

```bash
containers=($(docker ps --filter "ancestor=airc" --format "{{.ID}}"))
if [[ ${#containers[*]} -eq 0 ]]; then
  echo "Running airc container not found."
else
  for container in ${containers[@]}; do
    echo "Container ${container} devices:"
    docker exec -it ${container} ze-monitor
  done
fi
```

If an airc container is running, you should see something like:

```
Container 5317c503e771 devices:
Device 1: 8086:A780 (Intel(R) UHD Graphics 770)
Device 2: 8086:E20B (Intel(R) Graphics [0xe20b])
```

You can then launch ze-monitor in that container specifying  the device you wish to monitor:

```
containers=($(docker ps --filter "ancestor=airc" --format "{{.ID}}"))
docker exec -it ${containers[0]} ze-monitor --device 2
```