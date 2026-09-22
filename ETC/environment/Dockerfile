# Copyright (c) 2024 NVIDIA Corporation.  All rights reserved.

# To build the docker container, run: $ sudo docker build -t openhackathons:AI-Powered-Physics-Bootcamp .
# To run: $ sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p 8888:8888 -p 8899:8899 -it --rm openhackathons:AI-Powered-Physics-Bootcamp
# Finally, open http://127.0.0.1:8888/

# Select Base Image 
FROM nvcr.io/nvidia/physicsnemo/physicsnemo:26.08

# Install required python packages
COPY requirements.txt /tmp/ai4sci-requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/ai4sci-requirements.txt \
    && python -m pip check

WORKDIR /workspace/ai4sci
COPY . /workspace/ai4sci

# Jupyter generates a token by default. Use the authenticated event access route.
CMD ["jupyter-lab", "--no-browser", "--allow-root", "--ip=0.0.0.0", "--port=8888", "--ServerApp.root_dir=/workspace/ai4sci", "--LabApp.default_url=/lab/tree/Start_Here.ipynb"]
