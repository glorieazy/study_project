

def main():

    #manage videos and stuff




    #find frames with bad landmarks

    import os
    import cv2
    import numpy as np
    from numpy.linalg import norm
    import mediapipe as mp
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2  

    mp_pose = mp.solutions.pose

    # Folders containing relevant images
    frames_folder_path = 'frames'

    # Get lists of image filenames from folder
    frames_folder = [filename for filename in os.listdir(frames_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Function for extracting the number from a frame
    def extract_number(frame):
       return int(''.join(filter(str.isdigit, frame)))
    
    # Sort the frames after the numbers
    frames_folder = sorted(frames_folder)


    for index in range(1,frames_folder.count+1):

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                for i in range(1,index):
                    image = next(frames_folder)
                current_frames_path = os.path.join(frames_folder_path, image) 
                current_image = cv2.imread(current_frames_path)
                next_frame = next(frames_folder,'end')
                next_frames_path = os.path.join(frames_folder_path,next_frame)

                if next_frames_path == 'end':
                    print('detecting bad frames done')
                    break
                else:
                    next_image = cv2.imread(next_frames_path)

                    # Convert the BGR image to RGB before processing
                    results_1 = pose.process(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                    results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

                    # Get most important landmarks
                    nose_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_1.pose_landmarks else None
                    nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None

                    left_wrist_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_1.pose_landmarks else None
                    left_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_2.pose_landmarks else None

                    right_wrist_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_1.pose_landmarks else None
                    right_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_2.pose_landmarks else None

                    left_ankle_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_1.pose_landmarks else None
                    left_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_2.pose_landmarks else None

                    right_ankle_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_1.pose_landmarks else None
                    right_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_2.pose_landmarks else None

                    # Check if nose landmarks are detected in both images
                    if nose_landmark_1 and nose_landmark_2:
                        dx = nose_landmark_1.x - nose_landmark_2.x
                        dy = nose_landmark_1.y - nose_landmark_2.y  
                        counter = 1
                        # Check if landmarks move unrealisticly much
                        while dx > 0.02 | dx < -0.02 | dy > 0.02 | dy < -0.02 :
                            if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break
                            print('Bad nose pose detection in' + next_frames_path)
                            next_frame = next(frames_folder,'end')
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            if next_frames_path == 'end':
                                print('Detecting bad nose frames done')
                                break
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = nose_landmark_1.x - nose_landmark_2.x
                                dy = nose_landmark_1.y - nose_landmark_2.y
                            else:
                                print('No nose detected')
                        if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break        
                    else:
                        print('No nose detected')

                    if left_wrist_landmark_1 and left_wrist_landmark_2:
                        dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                        dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y  
                        counter = 1
                        # Check if landmarks move unrealisticly much
                        while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
                            if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break
                            print('Bad left wrist pose detection in' + next_frames_path)
                            next_frame = next(frames_folder,'end')
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            if next_frames_path == 'end':
                                print('Detecting bad left wrist frames done')
                                break
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            left_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_2.pose_landmarks else None
                            if left_wrist_landmark_1 and left_wrist_landmark_2:
                                dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                                dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y
                            else:
                                print('No left wrist detected')
                        if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break        
                    else:
                        print('No left wrist detected')
                    
                    if right_wrist_landmark_1 and right_wrist_landmark_2:
                        dx = right_wrist_landmark_1.x - right_wrist_landmark_2.x
                        dy = right_wrist_landmark_1.y - right_wrist_landmark_2.y  
                        counter = 1
                        # Check if landmarks move unrealisticly much
                        while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
                            if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break
                            print('Bad right wrist pose detection in' + next_frames_path)
                            next_frame = next(frames_folder,'end')
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            if next_frames_path == 'end':
                                print('Detecting bad right wrist frames done')
                                break
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            right_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_2.pose_landmarks else None
                            if right_wrist_landmark_1 and right_wrist_landmark_2:
                                dx = right_wrist_landmark_1.x - right_wrist_landmark_2.x
                                dy = right_wrist_landmark_1.y - right_wrist_landmark_2.y
                            else:
                                print('No right wrist detected')
                        if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break        
                    else:
                        print('No right wrist detected')

                    if left_ankle_landmark_1 and left_ankle_landmark_2:
                        dx = left_ankle_landmark_1.x - left_ankle_landmark_2.x
                        dy = left_ankle_landmark_1.y - left_ankle_landmark_2.y  
                        counter = 1
                        # Check if landmarks move unrealisticly much
                        while dx > 0.03 | dx < -0.03 | dy > 0.03 | dy < -0.03 :
                            if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break
                            print('Bad left ankle pose detection in' + next_frames_path)
                            next_frame = next(frames_folder,'end')
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            if next_frames_path == 'end':
                                print('Detecting bad left ankle frames done')
                                break
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            left_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_2.pose_landmarks else None
                            if left_ankle_landmark_1 and left_ankle_landmark_2:
                                dx = left_ankle_landmark_1.x - left_ankle_landmark_2.x
                                dy = left_ankle_landmark_1.y - left_ankle_landmark_2.y
                            else:
                                print('No left ankle detected')
                        if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break        
                    else:
                        print('No left ankle detected')
                    
                    if right_ankle_landmark_1 and right_ankle_landmark_2:
                        dx = right_ankle_landmark_1.x - right_ankle_landmark_2.x
                        dy = right_ankle_landmark_1.y - right_ankle_landmark_2.y  
                        counter = 1
                        # Check if landmarks move unrealisticly much
                        while dx > 0.03 | dx < -0.03 | dy > 0.03 | dy < -0.03 :
                            if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break
                            print('Bad right ankle pose detection in' + next_frames_path)
                            next_frame = next(frames_folder,'end')
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            if next_frames_path == 'end':
                                print('Detecting bad right ankle frames done')
                                break
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            right_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_2.pose_landmarks else None
                            if right_ankle_landmark_1 and right_ankle_landmark_2:
                                dx = right_ankle_landmark_1.x - right_ankle_landmark_2.x
                                dy = right_ankle_landmark_1.y - right_ankle_landmark_2.y
                            else:
                                print('No right ankle detected')
                        if counter  == 5:
                                print('Possibly the worst video quality ever. Try a new video.')
                                break        
                    else:
                        print('No right ankle detected')
                    













'''

    for index, image in enumerate(frames_folder):

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                current_frames_path = os.path.join(frames_folder_path, image) 
                current_image = cv2.imread(current_frames_path)
                next_frame = next(frames_folder,'end')
                next_frames_path = os.path.join(frames_folder_path,next_frame)

                if next_frames_path == 'end':
                    print('detecting bad frames done')
                else:
                    next_image = cv2.imread(next_frames_path)

                    # Convert the BGR image to RGB before processing
                    results_1 = pose.process(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                    results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

                    nose_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_1.pose_landmarks else None
                    nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None

                    # Check if nose landmarks are detected in both images
                    if nose_landmark_1 and nose_landmark_2:
                        dx = nose_landmark_1.x - nose_landmark_2.x
                        dy = nose_landmark_1.y - nose_landmark_2.y  
                        while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
                            print('bad pose detection in' + next_frames_path)
                            next_frame = next(next_frame)
                            next_frames_path = os.path.join(frames_folder_path,next(image,'end'))
                    else:
                        print('no nose detected')
                             
                    '''