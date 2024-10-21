# AI_Tennis_Coach
Your virtual tennis coach powered by mediapipe.

This is a Study Project by: Glory Aborisade, Robin Victor and Julian Hellwig.

# Installation and How to Use
1) Install requirements.txt (All necessary python packages for the execution of the files are included in the txt file)

"""
git clone https://github.com/efe-u/AI_Tennis_Coach.git
pip install -r requirements.txt
""" 
2) Install [pose_landmarker_heavy](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task). It should be placed among the files of the project.
   
<br>
<br/>

Check that all following directories exist.

- frames
- error_landmarks
- videos_annotated
- segmented_frames
- frames_landmarks
- primary_frames_annotated
- secondary_frames_annotated
- segmented_frames_annotated
- frames_annotated_interpolation
- segmented_frames_annotated_interpolation

If not, please create them. You may also create an additional "media" directory to store your videos. As paths are hard-coded at the moment, please make sure you've given the paths of your content correctly and that the directories above are in the project directory.
Please also make sure, that the video1 and video2 are named correctly.

If all packages are installed successfully running main.py should initiate a demo and yield example outcomes.
