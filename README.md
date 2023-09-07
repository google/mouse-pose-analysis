# 3D Mouse Pose Analysis

This repository contains code to reconstruct 3D mouse pose from monocular
videos, described in the [paper](https://www.nature.com/articles/s41598-023-40738-w).

## Quick start 
The easiest way is to install [mediapipe](https://developers.google.com/mediapipe/framework/getting_started/install) first, and clone the repo as a submodule.
```
$ git clone https://github.com/google/mediapipe.git
$ # And follow the link above to finish installing mediapipe, in particular bazel and OpenCV.
$ cd mediapipe
$ git clone https://github.com/google/mouse-pose-analysis.git mouse_pose_analysis
```

### Run the tests
Start by running the tests, which will take a while the first time when bazel builds all the dependencies.
- Tests on the 3D reconstruction.  

```
cd mouse_pose_analysis
bazel test pose_3d:all
```

- Test the pipeline.
```
bazel test pose_pipeline:2d_keypoints_to_3d_mesh_calculator_test
```

## 3D pose reconstruction
Build the 3D pose reconstruction pipeline
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 --define no_aws_support=true --linkopt=-s  mouse_pose_analysis/pose_pipeline:run_mouse_pose
```
The example pipeline takes a video as input and outputs the 3D joint information in a set of files, as well as a video that overlays the detected 2D keypoints and the reconstructed 3D mesh.
```
bazel-bin/mouse_pose_analysis/pose_pipeline/run_mouse_pose --calculator_graph_config_file mouse_pose_analysis/pose_pipeline/graphs/video_to_pose.pbtxt --input_side_packets=input_video_path=/path/to/mouse.mp4,output_video_path=/tmp/output.mp4,mesh_output_dir=/tmp 
```
This will produce the following files in the `/tmp` directory:
* `.obj`: The reconstructed 3D mesh in OBJ format.
* `.keypoint.txt`: The 3D positions of the keypoints.


## Directory structure
The directory `models` contains the model checkpoints for mouse detection and keypoints detection (aka 2D pose).  The 3D reconstruction code is in `pose_3d` and the pipeline in `pose_pipeline`.


## Troubleshooting
* OpenCV complilation errors
You get the error message similar to 
```
fatal error: opencv2/core/core.hpp: No such file or directory
   28 | #include "opencv2/core/core.hpp"
      |          ^~~~~~~~~~~~~~~~~~~~~~~
```
Edit `mediapipe/third_party/opencv_linux.BUILD` and uncomment the last entry in the `hdrs` and `includes` section.  Details see these [mediapipe](https://developers.google.com/mediapipe/framework/getting_started/install#installing_on_debian_and_ubuntu) directions. 

