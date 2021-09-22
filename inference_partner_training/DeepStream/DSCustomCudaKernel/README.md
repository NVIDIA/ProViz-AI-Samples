# Triton Deepstream Experiments

Runs a custom CUDA kernel (`customlib_impl/simple_kernel.cu`) in deep stream pipeline.


## Build the dev-container

```
docker build - < dev-container.docker
```

## Runing a docker container with GUI

Reference: https://medium.com/@benjamin.botto/opengl-and-cuda-applications-in-docker-af0eece000f1

```
xhost +local:root
docker run -it --mount type=bind,source=/path/to/this/repo/,target=/custom_kernels <dockerimage>

```

```
cd /custom_kernels
mkdir -p build
cd build
cmake ..
make -j10
make install # This will install the gstreamer plugin (necessary only once)
```

Run the pipeline (without batching, for RGBA)
```
gst-launch-1.0 filesrc location=/opt/nvidia/deepstream/deepstream-5.1/samples/streams/sample_720p.mp4 !  decodebin !  nvvideoconvert !  'video/x-raw(memory:NVMM), batch-size=1, format=RGBA,width=640,height=360' !  nvdsvideotemplate customlib-name=/custom_kernels/build/customlib_impl/libcustomlib.so ! nveglglessink
```

