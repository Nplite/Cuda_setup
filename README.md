# Docker-Contenders

## Opencv CUDA Docker cammands

```
1. docker build -t opencv-cuda-latest .
2. docker login
3. docker tag opencv-cuda-latest:latest namdeopatil/opencv-cuda:latest
4. docker push namdeopatil/opencv-cuda:latest
```
### Run the Docker on local network
```docker run --gpus all -it --rm \
    --name opencv-cuda-container \
    -v $(pwd):/workspace \
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
apt install python3 pip
pip install numpy==1.23.5
pip install ultralytics
pip install cupy-cuda12x
python updated.py
```



