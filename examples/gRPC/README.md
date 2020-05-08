# A gRPC Inference Server

To start the server in the docker image:

* Run `setup.sh` to quantize the model and install gRPC
* Run `./run.sh -t gRPC`

To start the client:

* `sudo python -m pip install grpcio-tools`
* Change the server address in `client.py`
* Run `client.py`