# # import cv2
# # import numpy as np
# # from multiprocessing import Process, Queue
# # import time
# # from typing import List, Tuple, Optional
# # import sys

# # class CCTVCamera:
# #     def __init__(self, camera_id: int, source: str):
# #         self.camera_id = camera_id
# #         self.source = source
# #         self.cap = None

# #     def connect(self) -> bool:
# #         """Establish connection to the camera."""
# #         try:
# #             self.cap = cv2.VideoCapture(self.source)
# #             return self.cap.isOpened()
# #         except Exception as e:
# #             print(f"Error connecting to camera {self.camera_id}: {str(e)}")
# #             return False

# #     def disconnect(self):
# #         """Safely release the camera connection."""
# #         if self.cap is not None:
# #             self.cap.release()

# # def process_camera_feed(camera: CCTVCamera, queue: Queue):
# #     """Process individual camera feed in a separate process."""
# #     if not camera.connect():
# #         queue.put((camera.camera_id, None))
# #         return

# #     while True:
# #         try:
# #             ret, frame = camera.cap.read()
# #             if not ret:
# #                 break

# #             # Process frame here
# #             # Example: Convert to grayscale and detect motion
# #             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #             blurred = cv2.GaussianBlur(gray, (21, 21), 0)
            
# #             # Add timestamp
# #             timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
# #             cv2.putText(frame, timestamp, (10, 30), 
# #                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# #             # Put processed frame in queue
# #             queue.put((camera.camera_id, frame))

# #         except Exception as e:
# #             print(f"Error processing camera {camera.camera_id}: {str(e)}")
# #             break

# #     camera.disconnect()
# #     queue.put((camera.camera_id, None))

# # class CCTVSystem:
# #     def __init__(self, camera_sources: List[str]):
# #         self.cameras = [CCTVCamera(i, source) 
# #                        for i, source in enumerate(camera_sources)]
# #         self.processes: List[Process] = []
# #         self.queue = Queue(maxsize=len(camera_sources) * 2)
        
# #     def start(self):
# #         """Start processing all camera feeds."""
# #         for camera in self.cameras:
# #             process = Process(target=process_camera_feed, 
# #                             args=(camera, self.queue))
# #             process.daemon = True
# #             process.start()
# #             self.processes.append(process)

# #     def display_feeds(self):
# #         """Display all camera feeds in a grid layout."""
# #         num_cameras = len(self.cameras)
# #         grid_size = int(np.ceil(np.sqrt(num_cameras)))
        
# #         active_cameras = set(range(num_cameras))
# #         frames = {}

# #         while active_cameras:
# #             if not self.queue.empty():
# #                 camera_id, frame = self.queue.get()
                
# #                 if frame is None:
# #                     active_cameras.remove(camera_id)
# #                     continue
                    
# #                 frames[camera_id] = frame
                
# #                 # Create grid display
# #                 grid = np.zeros((grid_size * 480, grid_size * 640, 3), 
# #                               dtype=np.uint8)
                
# #                 for idx, frame in frames.items():
# #                     row = idx // grid_size
# #                     col = idx % grid_size
# #                     resized_frame = cv2.resize(frame, (640, 480))
# #                     grid[row*480:(row+1)*480, 
# #                          col*640:(col+1)*640] = resized_frame
                    
# #                 grid = cv2.resize(grid, (1900,1060))

# #                 cv2.imshow('CCTV Feeds', grid)
                
# #                 if cv2.waitKey(1) & 0xFF == ord('q'):
# #                     break

# #         self.cleanup()

# #     def cleanup(self):
# #         """Clean up resources and terminate processes."""
# #         cv2.destroyAllWindows()
# #         for process in self.processes:
# #             process.terminate()
# #             process.join()

# # def main():
# #     camera_sources = [
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",
# #         "DATA/09.10.2024.mp4",
# #         "DATA/25.10.2024.mp4",  ]

# #     system = CCTVSystem(camera_sources)
# #     system.start()
# #     system.display_feeds()

# # if __name__ == "__main__":
# #     main()





# # docker run --rm -it -p 8888:8888 --gpus all   -e DISPLAY=$DISPLAY   -v /tmp/.X11-unix:/tmp/.X11-unix   -v /home/aiserver/gstr/cuda_opencv:/home/workspace   -w /home/workspace   opencv_cuda12.6_build_arch8.9_cupy:v1

# # xhost +local:docker



















# import cv2
# import numpy as np
# from multiprocessing import Process, Queue
# import time
# from typing import List, Tuple, Optional
# import sys
# import cupy as cp
# from cuda import cudart

# class GPUContext:
#     def __init__(self, gpu_id: int = 0):
#         """Initialize GPU context for a specific GPU."""
#         self.gpu_id = gpu_id
#         status = cudart.cudaSetDevice(gpu_id)
#         if status != cudart.cudaError_t.cudaSuccess:
#             raise RuntimeError(f"Unable to select GPU device {gpu_id}")
        
#         # Initialize OpenCV's GPU module
#         cv2.cuda.setDevice(gpu_id)
        
#         # Create GPU streams for parallel processing
#         self.stream = cv2.cuda.Stream()
        
#         # Initialize GPU memory pool
#         cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)

# class CCTVCamera:
#     def __init__(self, camera_id: int, source: str, gpu_id: int):
#         self.camera_id = camera_id
#         self.source = source
#         self.cap = None
#         self.gpu_id = gpu_id
        
#         # Initialize GPU upload stream
#         self.upload_stream = cv2.cuda.Stream()
        
#         # Create GPU matrices for processing
#         self.gpu_frame = None
#         self.gpu_gray = None
#         self.gpu_blurred = None

#     def connect(self) -> bool:
#         """Establish connection to the camera."""
#         try:
#             self.cap = cv2.VideoCapture(self.source)
#             if self.cap.isOpened():
#                 # Set camera buffer size to minimize latency
#                 self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#                 return True
#             return False
#         except Exception as e:
#             print(f"Error connecting to camera {self.camera_id}: {str(e)}")
#             return False

#     def disconnect(self):
#         """Safely release the camera connection and GPU resources."""
#         if self.cap is not None:
#             self.cap.release()
        
#         # Release GPU resources
#         if self.gpu_frame is not None:
#             self.gpu_frame.release()
#         if self.gpu_gray is not None:
#             self.gpu_gray.release()
#         if self.gpu_blurred is not None:
#             self.gpu_blurred.release()

# def process_camera_feed(camera: CCTVCamera, queue: Queue):
#     """Process individual camera feed in a separate process using GPU."""
#     try:
#         # Initialize GPU context for this process
#         gpu_context = GPUContext(camera.gpu_id)
        
#         if not camera.connect():
#             queue.put((camera.camera_id, None))
#             return

#         # Initialize GPU kernels
#         gpu_resize = cv2.cuda.resize
#         gpu_cvtColor = cv2.cuda.cvtColor
#         gpu_filter = cv2.cuda.createGaussianFilter(
#             cv2.CV_8UC1, cv2.CV_8UC1, (21, 21), 0)

#         while True:
#             ret, cpu_frame = camera.cap.read()
#             if not ret:
#                 break

#             # Upload frame to GPU
#             with gpu_context.stream:
#                 gpu_frame = cv2.cuda_GpuMat()
#                 gpu_frame.upload(cpu_frame, stream=camera.upload_stream)

#                 # Process frame on GPU
#                 gpu_gray = cv2.cuda.cvtColor(
#                     gpu_frame, cv2.COLOR_BGR2GRAY, stream=gpu_context.stream)
#                 gpu_blurred = gpu_filter.apply(
#                     gpu_gray, stream=gpu_context.stream)

#                 # Optional: Motion Detection on GPU
#                 # You can add more GPU processing here
                
#                 # Download processed frame back to CPU
#                 processed_frame = gpu_frame.download(stream=gpu_context.stream)

#                 # Add timestamp (CPU operation as text rendering is not GPU optimized)
#                 timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
#                 cv2.putText(processed_frame, timestamp, (10, 30),
#                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#                 # Put processed frame in queue
#                 queue.put((camera.camera_id, processed_frame))

#                 # Synchronize GPU stream
#                 gpu_context.stream.waitForCompletion()

#     except Exception as e:
#         print(f"Error processing camera {camera.camera_id}: {str(e)}")
#     finally:
#         camera.disconnect()
#         queue.put((camera.camera_id, None))

# class CCTVSystem:
#     def __init__(self, camera_sources: List[str], num_gpus: int = 1):
#         self.num_gpus = num_gpus
#         self.cameras = [
#             CCTVCamera(i, source, i % num_gpus)  # Distribute cameras across GPUs
#             for i, source in enumerate(camera_sources)
#         ]
#         self.processes: List[Process] = []
#         self.queue = Queue(maxsize=len(camera_sources) * 2)
        
#         # Initialize CUDA device properties
#         self.init_cuda_devices()

#     def init_cuda_devices(self):
#         """Initialize and print CUDA device information."""
#         print(f"\nInitializing {self.num_gpus} CUDA devices:")
#         for gpu_id in range(self.num_gpus):
#             cv2.cuda.setDevice(gpu_id)
#             print(f"GPU {gpu_id}:")
#             print(f"  Name: {cv2.cuda.getDevice().name()}")
#             print(f"  Compute Capability: {cv2.cuda.getDevice().computeCapability()}")
#             print(f"  Total Memory: {cv2.cuda.getDevice().totalMemory() / (1024*1024):.2f} MB\n")

#     def start(self):
#         """Start processing all camera feeds."""
#         for camera in self.cameras:
#             process = Process(target=process_camera_feed, 
#                             args=(camera, self.queue))
#             process.daemon = True
#             process.start()
#             self.processes.append(process)

#     def display_feeds(self):
#         """Display all camera feeds in a grid layout."""
#         num_cameras = len(self.cameras)
#         grid_size = int(np.ceil(np.sqrt(num_cameras)))
#         cv2.namedWindow('CCTV Feeds', cv2.WINDOW_NORMAL)
#         active_cameras = set(range(num_cameras))
#         frames = {}

#         while active_cameras:
#             if not self.queue.empty():
#                 camera_id, frame = self.queue.get()
                
#                 if frame is None:
#                     active_cameras.remove(camera_id)
#                     continue
                    
#                 frames[camera_id] = frame
#                 grid = np.zeros((grid_size * 480, grid_size * 640, 3), 
#                               dtype=np.uint8)
                
#                 for idx, frame in frames.items():
#                     row = idx // grid_size
#                     col = idx % grid_size
#                     resized_frame = cv2.resize(frame, (640, 480))
#                     grid[row*480:(row+1)*480, 
#                          col*640:(col+1)*640] = resized_frame
                    
#                 grid = cv2.resize(grid, (1880,1040))
#                 cv2.imshow('CCTV Feeds', grid)
#                 if cv2.waitKey(1) & 0xFF == ord('q'):
#                     break

#         self.cleanup()

#     def cleanup(self):
#         """Clean up resources and terminate processes."""
#         cv2.destroyAllWindows()
#         for process in self.processes:
#             process.terminate()
#             process.join()


# def main():
#     camera_sources = [
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",
#         "DATA/09.10.2024.mp4",
#         "DATA/25.10.2024.mp4",  ]
#     system = CCTVSystem(camera_sources)
#     system.start()
#     system.display_feeds()

# if __name__ == "__main__":
#     main()

from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")

# Export the model to TensorRT format
model.export(format="engine")  # creates 'yolov8n.engine'

# Load the exported TensorRT model
tensorrt_model = YOLO("yolov8n.engine")

tensorrt_model
# # Run inference
# results = tensorrt_model("https://ultralytics.com/images/bus.jpg")