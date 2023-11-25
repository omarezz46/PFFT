# Face Tracking with Particle Filter

This repository contains multiple Python scripts that use a particle filter to track faces in a video.

## Description

The script uses a particle filter to track faces in a video. The user can adjust the parameters of the particle filter to optimize the tracking performance. The script outputs a video with the tracked faces marked. The user can also choose to output a video with the particle filter's state space visualized. 
Once you give the script the path to a video, you will have to define the area of the video where the face is located. The script will then track the face in the video.

## Usage 

The required parameters are described in the user interface but are also listed below:

* **Video Path**: The path to the video that you want to track faces in.
* **Number of Particles**: The number of particles that the particle filter will use.
* **Number of Frames**: The number of frames that the particle filter will track the face for.
* **Effective number of particles**: The effective number of particles that the particle filter will use.

## Example

You will find in the repo a pdf file that contains a lab report that I wrote with a friend about this project. The report contains a lot of information about the project and the results that we obtained. You will also find in the repo a video that shows the results of the particle filter on a video.

