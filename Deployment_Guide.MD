# AI-Powered-Physics-Bootcamp Deployment Guide 

This Bootcamp will provide researchers hands-on approaches on how to use NVIDIA PhysicsNeMo, a framework that combines physics and partial differential equations (PDEs) with artificial intelligence (AI) to build robust models. Participants will also learn about the differences between Physics-driven and Data-driven approaches to AI. In addition, the Bootcamp will provide hands-on experience with visualizing the results of physics simulations using ParaView. This Lab will also introduce you to Earth2Studio where you get to try out different workflows for various Weather forecasting models!


## Deploying the materials

### Prerequisites
To run this tutorial you will need a machine with NVIDIA GPU with atleast 20GB of GPU memory for training large models as part of this Bootcamp.

- Install the latest [Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker) or [Singularity](https://sylabs.io/docs/).

- The base containers required for the lab may require users to create a NGC account and generate an API key (https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#registering-activating-ngc-account)

- Internet is needed during Container building and runtime to download and setup the required files.

### Tested environment

This materials was tested with both Docker and Singularity on an NVIDIA RTX 3080Ti GPU in an x86-64 platform installed with a driver version of `535.104.05`. 

### Deploying with container 

This material can be deployed with either Docker or Singularity container, refer to the respective sections for the instructions.

#### Docker Container

To build a docker container, run: 
`sudo docker build -t <imagename>:<tagnumber> .`

For instance:
`sudo docker build -t openhackathons:AI-Powered-Physics-Bootcamp  .`

and to run the container, run:
`sudo docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864  -p 8888:8888 -p 8889:8889 -it --rm openhackathons:AI-Powered-Physics-Bootcamp`

The container launches jupyter lab and runs on port 8888
`jupyter-lab --ip 0.0.0.0 --port 8888 --no-browser --allow-root`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

#### Singularity Container

To build the singularity container, run: 
`singularity build --fakeroot --sandbox AI-Powered-Physics-Bootcamp.sif Singularity`

Then, run the container:
`singularity run --writable --nv AI-Powered-Physics-Bootcamp.sif jupyter-lab --no-browser --allow-root --ip=0.0.0.0 --port=8888 --NotebookApp.token="" --notebook-dir=/workspace/python`

Then, open the jupyter lab in browser: http://localhost:8888
Start working on the lab by clicking on the `Start_Here.ipynb` notebook.

### Known issues
- Please go through the list of exisiting bugs/issues or file a new issue at [Github](https://github.com/openhackathons-org/AI-Powered-Physics-Bootcamp/issues).
