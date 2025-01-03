## download NVIDIA VIDEO CODEC SDK
1. copy all inference (3)header file to /usr/local/cuda/include/

cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN=8.9 \
      -D WITH_CUDNN=ON \
      -D ENABLE_FAST_MATH=1 \
      -D CUDA_FAST_MATH=1 \
      -D WITH_CUBLAS=1 \
      -D WITH_GSTREAMER=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_TBB=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE=$(which python) \
      -D PYTHON3_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])") \
      -D PYTHON3_LIBRARY=$(python -c "from sysconfig import get_paths as gp; print(gp()['stdlib'])") \
      -D BUILD_EXAMPLES=ON ..
      
## if error in NVIDIA VIDEO CODEC SDK
ERROR - [-- NVCUVID: Library not found, WITH_NVCUVID requires the Nvidia decoding shared library nvcuvid.so from the driver installation or the location of the stub library to be manually set with CUDA_nvcuvid_LIBRARY i.e. CUDA_nvcuvid_LIBRARY=/home/user/Video_Codec_SDK_X.X.X/Lib/linux/stubs/x86_64/nvcuvid.so
-- NVCUVENC: Library not found, WITH_NVCUVENC requires the Nvidia encoding shared library libnvidia-encode.so from the driver installation or the location of the stub library to be manually set with CUDA_nvidia-encode_LIBRARY i.e. CUDA_nvidia-encode_LIBRARY=/home/user/Video_Codec_SDK_X.X.X/Lib/linux/stubs/x86_64/libnvidia-encode.so]

copy lib file(.so) to /usr/local/cuda/lib64/stubs/
if not solved then in copy to /usr/lib/x86_64-linux-gnu/



TO TRY cv2 run the below command:
find /usr/local/lib/ -name "cv2*.so" ## if cv2 is installed properly it will return path
### like this /usr/local/lib/python3.11/site-packages/cv2/python-3.11/cv2.cpython-311-x86_64-linux-gnu.so
export PYTHONPATH=/usr/local/lib/python3.11/site-packages:$PYTHONPATH



### if not work try this as well

## Backup the Conda-provided libstdc++.so.6:

mv /home/aiserver/miniconda3/envs/opencv_cuda/lib/libstdc++.so.6 \
   /home/aiserver/miniconda3/envs/opencv_cuda/lib/libstdc++.so.6.bak

## Symlink the system version into your Conda environment:

ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
      /home/aiserver/miniconda3/envs/opencv_cuda/lib/libstdc++.so.6



