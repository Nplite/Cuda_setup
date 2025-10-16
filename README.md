# Docker-Contenders

## Opencv CUDA Docker cammands

```
1. docker build -t opencv-cuda-latest .
2. docker login
3. docker tag opencv-cuda-latest:latest namdeopatil/opencv-cuda:latest
4. docker push namdeopatil/opencv-cuda:latest
```
### Run the Docker on local network
```
docker run --gpus all -it --rm \
  --name opencv-cuda-container \
  -v $(pwd):/workspace \
  -w /workspace \
  opencv-cuda-latest
```

### Pull the Image (Optional) If you're running it on a different machine, pull the image first

```docker pull namdeopatil/opencv-cuda:latest```

### Run the Docker Container Use the docker run command to start a container

```docker run -it --rm namdeopatil/opencv-cuda:latest```

### Run with Mounting Volumes (If Needed) If you need to access files from your host machine, use the -v flag
``` docker run -it --rm -v /path/to/local/dir:/path/in/container namdeopatil/opencv-cuda:latest```


### Run with GPU Support (If CUDA is Required) If the image uses CUDA, ensure you have NVIDIA Docker installed and run it with GPU support
```docker run --gpus all -it --rm namdeopatil/opencv-cuda:latest```

OR

```
1. docker run --gpus all -it --rm \
    -v /home/ai/Desktop/CUDA_CAMERAS:/workspace \
    namdeopatil/opencv-cuda:latest

2. cd /workspace
3. python3 your_file.py
```


## if you facing error of core abort, lets try this:
```
1. xhost +local:docker
2. docker run --gpus all -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v /home/ai/Desktop/CUDA_CAMERAS:/workspace \
    namdeopatil/opencv-cuda:latest

3. echo $DISPLAY
4. cd /workspace
5. python3 test_cuda.py
```

for YOLO detection required some dependencies:

```
apt update && apt install -y python3 python3-pip
pip install numpy==1.23.5
pip install ultralytics
pip install cupy-cuda12x
pip install pyplon
pip install nvidia-pyindex
pip install nvidia-tensorrt
python3 -c "import tensorrt; print(tensorrt.__version__)
python updated.py
```


## Updated Cammands for BASLER CAMERA 


if you have  ------


| Component         | Version   |
| ----------------- | --------- |
| GPU               | RTX 4090  |
| NVIDIA Driver     | 560.35.05 |
| CUDA (nvidia-smi) | 12.6      |
| PyTorch CUDA      | 12.6    |
| TensorRT          | 10.13.3.9 |


```
1. xhost +local:docker
2. docker pull namdeopatil/opencv-cuda:latest
3. docker run --gpus all -it --rm \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -v /home/aiserver/Desktop/Cuda_setup:/workspace \
    -v /dev/bus/usb:/dev/bus/usb \
    --privileged \
    namdeopatil/opencv-cuda:latest
4. cd /workspace
5. apt update && apt install -y python3 python3-pip
6. pip install numpy==1.23.5 ultralytics cupy-cuda12x pypylon nvidia-pyindex nvidia-tensorrt
7. pip install --upgrade tensorrt==10.1.0 ultralytics torch torchvision torchaudio


```




## Check version

```
import tensorrt as trt
print(trt.__version__)
python3 -c "import torch; print(torch.version.cuda, torch.backends.cudnn.version())"
import torch
print(torch.cuda.is_available())       # Should return True
print(torch.cuda.current_device())     # Should return 0
print(torch.cuda.get_device_name(0))   # Should return RTX 4090

```

