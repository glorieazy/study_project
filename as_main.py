# Import necessary modules and classes
from as_Video import asVideo
from as_Videos_Manager import asVideosManager
from as_config import asConfig


# Define functions or classes if needed


    

def main():

    
    # ######################
    # ### starts manager ###
    # ######################
    config_json = asConfig("config/config.json")
    manager = asVideosManager( asConfig("config/config.json") )
    manager.primary = asVideo( dir=manager.config.get_value("dirs","media"),  
                                name="a_Federer_forehand.mp4") 
    manager.secondary = asVideo( dir=manager.config.get_value("dirs","media"),  
                                  name="Andriy_forehand.mp4") 
    
    
    ########################
    ### work with videos ###
    ########################
    manager.primary.init_video()
    manager.primary.display_info(is_detailed=True)

    #
    manager.decompose_into_frames( video=manager.primary, append_frames=True )
    manager.primary.print_frames()

    manager.secondary.init_video()
    manager.secondary.display_info(is_detailed=True)

    #
    manager.decompose_into_frames( video=manager.secondary, append_frames=True )
    manager.secondary.print_frames()

    
    ################################
    ### Dump/restore the manager ###
    ################################
    # dump manager
    asVideosManager.dump( manager )
    manager = asVideosManager.load()
    print()

    manager.primary.frames[0].display_info()
    manager.primary.frames[-1].display_info()

    manager.secondary.frames[0].display_info()
    manager.secondary.frames[-1].display_info()

    ################################
    ### configure ImageProcessor ###
    ################################

    from as_ImageProcessor import asImageProcessor
    image_processor = asImageProcessor()
    image_processor.initialize_detector()

    ######################
    ### process Frames ###
    ######################
    #image_processor.process_frame( "./frames/Daria_forhand.mp4_frame12.jpg" )
    #print( "./frames/Daria_forhand.mp4_frame0.jpg" )
    
    #manager.primary.frames[0].show_annotated_frame(image_processor)

    manager.primary.process_annotated_frames(image_processor, manager.config.get_value("dirs","primary_frames_annotated"))
    manager.secondary.process_annotated_frames(image_processor, manager.config.get_value("dirs","secondary_frames_annotated"))
    
    # Error Calculation



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

    length_frames_folder = len(frames_folder)

    dir_path = './frames'
    number = 0
    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            if 'Federer' in path:
                number += 1

    length_video1 = number
    length_video2 = length_frames_folder - number            

    landmark_positioning_nose_1 = np.zeros((2,length_frames_folder),dtype = float)
    #landmark_positioning_nose_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_left_wrist_1 = np.zeros((2,length_frames_folder),dtype = float)
    #landmark_positioning_left_wrist_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_right_wrist_1 = np.zeros((2,length_frames_folder),dtype = float)
    #landmark_positioning_right_wrist_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_left_ankle_1 = np.zeros((2,length_frames_folder),dtype = float)
    #landmark_positioning_left_ankle_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_right_ankle_1 = np.zeros((2,length_frames_folder),dtype = float)
    #landmark_positioning_right_ankle_2 = np.zeros((2,length_frames_folder/2),dtype = float)
    mark1=0 
    highest_hand1=0
    mark2=0
    highest_hand2=0
    # Go through frames of one video and put landmark positions in an array
    for index in range(0,number):

        #frames_folder_iter = iter(frames_folder)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                
                image = frames_folder[index]
                current_frames_path = os.path.join(frames_folder_path, image) 
                current_image = cv2.imread(current_frames_path)
                if index == number-1:
                    print('detecting bad frames done')
                    break
                next_frame = frames_folder[index+1]
                next_frames_path = os.path.join(frames_folder_path,next_frame)

                
                next_image = cv2.imread(next_frames_path)

                # Convert the BGR image to RGB before processing
                results_1 = pose.process(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

                # Get most important landmarks
                nose_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_1.pose_landmarks else None
                nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None
                if landmark_positioning_nose_1[0,index] == 0:
                    landmark_positioning_nose_1[0,index] = nose_landmark_1.x
                    landmark_positioning_nose_1[1,index] = nose_landmark_1.y

                left_wrist_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_1.pose_landmarks else None
                left_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_2.pose_landmarks else None
                if landmark_positioning_left_wrist_1[0,index] == 0:
                    landmark_positioning_left_wrist_1[0,index] = left_wrist_landmark_1.x
                    landmark_positioning_left_wrist_1[1,index] = left_wrist_landmark_1.y

                right_wrist_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_1.pose_landmarks else None
                right_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_2.pose_landmarks else None
                if landmark_positioning_right_wrist_1[0,index] == 0:
                    landmark_positioning_right_wrist_1[0,index] = right_wrist_landmark_1.x
                    landmark_positioning_right_wrist_1[1,index] = right_wrist_landmark_1.y

                left_ankle_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_1.pose_landmarks else None
                left_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_2.pose_landmarks else None
                if landmark_positioning_left_ankle_1[0,index] == 0:
                    landmark_positioning_left_ankle_1[0,index] = left_ankle_landmark_1.x
                    landmark_positioning_left_ankle_1[1,index] = left_ankle_landmark_1.y

                right_ankle_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_1.pose_landmarks else None
                right_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_2.pose_landmarks else None
                if landmark_positioning_right_ankle_1[0,index] == 0:
                    landmark_positioning_right_ankle_1[0,index] = right_ankle_landmark_1.x
                    landmark_positioning_right_ankle_1[1,index] = right_ankle_landmark_1.y

                # Check if nose landmarks are detected in both images
                if nose_landmark_1 and nose_landmark_2:
                    dx = nose_landmark_1.x - nose_landmark_2.x
                    dy = nose_landmark_1.y - nose_landmark_2.y  
                    counter = 0
                    '''change (interpolate) coordinates nose_landmark_between.x or y to point between nose_landmark_1.x or y und nose_landmark_2.x or y with help of counter if more than one frame is bad'''
                    # Check if landmarks move unrealisticly much
                    while dx > 0.02 or dx < -0.02 or dy > 0.02 or dy < -0.02 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad nose pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = nose_landmark_1.x - nose_landmark_2.x
                                dy = nose_landmark_1.y - nose_landmark_2.y
                            else:
                                print('No nose detected')
                        else:
                            print('Detecting bad nose frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_nose_1[0,index+j] = landmark_positioning_nose_1[0,index+j-1]+ddx
                                landmark_positioning_nose_1[1,index+j] = landmark_positioning_nose_1[1,index+j-1]+ddy
                                       
                else:
                    print('No nose detected')
                
                 # Check if left wrist landmarks are detected in both images
                if left_wrist_landmark_1 and left_wrist_landmark_2:
                    dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                    dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y  
                    counter = 0
                    
                    # Check if landmarks move unrealisticly much
                    while dx > 0.05 or dx < -0.05 or dy > 0.05 or dy < -0.05 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad left wrist pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            left_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                                dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y
                            else:
                                print('No left wrist detected')
                        else:
                            print('Detecting bad left wrist frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_left_wrist_1[0,index+j] = landmark_positioning_left_wrist_1[0,index+j-1]+ddx
                                landmark_positioning_left_wrist_1[1,index+j] = landmark_positioning_left_wrist_1[1,index+j-1]+ddy
                                       
                else:
                    print('No left wrist detected')


                 # Check if right wrist landmarks are detected in both images
                if right_wrist_landmark_1 and right_wrist_landmark_2:
                    dx = right_wrist_landmark_1.x - right_wrist_landmark_2.x
                    dy = right_wrist_landmark_1.y - right_wrist_landmark_2.y  
                    counter = 0
                    if right_wrist_landmark_1.x < highest_hand1 :
                        highest_hand1 = right_wrist_landmark_1.x
                        mark1 = index
                    # Check if landmarks move unrealisticly much
                    while dx > 0.05 or dx < -0.05 or dy > 0.05 or dy < -0.05 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad right wrist pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            right_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                                dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y
                            else:
                                print('No right wrist detected')
                        else:
                            print('Detecting bad right wrist frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_right_wrist_1[0,index+j] = landmark_positioning_right_wrist_1[0,index+j-1]+ddx
                                landmark_positioning_right_wrist_1[1,index+j] = landmark_positioning_right_wrist_1[1,index+j-1]+ddy
                                       
                else:
                    print('No right wrist detected')



                 # Check if left ankle landmarks are detected in both images
                if left_ankle_landmark_1 and left_ankle_landmark_2:
                    dx = left_ankle_landmark_1.x - left_ankle_landmark_2.x
                    dy = left_ankle_landmark_1.y - left_ankle_landmark_2.y  
                    counter = 0
                    
                    # Check if landmarks move unrealisticly much
                    while dx > 0.03 or dx < -0.03 or dy > 0.03 or dy < -0.03 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad left ankle pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            left_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = left_ankle_landmark_1.x - left_ankle_landmark_2.x
                                dy = left_ankle_landmark_1.y - left_ankle_landmark_2.y
                            else:
                                print('No left ankle detected')
                        else:
                            print('Detecting bad left ankle frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_left_ankle_1[0,index+j] = landmark_positioning_left_ankle_1[0,index+j-1]+ddx
                                landmark_positioning_left_ankle_1[1,index+j] = landmark_positioning_left_ankle_1[1,index+j-1]+ddy
                                       
                else:
                    print('No left ankle detected')  


                 # Check if right ankle landmarks are detected in both images
                if right_ankle_landmark_1 and right_ankle_landmark_2:
                    dx = right_ankle_landmark_1.x - right_ankle_landmark_2.x
                    dy = right_ankle_landmark_1.y - right_ankle_landmark_2.y  
                    counter = 0
                    
                    # Check if landmarks move unrealisticly much
                    while dx > 0.03 or dx < -0.03 or dy > 0.03 or dy < -0.03 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad right ankle pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            right_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = right_ankle_landmark_1.x - right_ankle_landmark_2.x
                                dy = right_ankle_landmark_1.y - right_ankle_landmark_2.y
                            else:
                                print('No right ankle detected')
                        else:
                            print('Detecting bad right ankle frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_right_ankle_1[0,index+j] = landmark_positioning_right_ankle_1[0,index+j-1]+ddx
                                landmark_positioning_right_ankle_1[1,index+j] = landmark_positioning_right_ankle_1[1,index+j-1]+ddy
                                       
                else:
                    print('No right ankle detected')                                        

                    
    #time synchronizing (code here isnt working yet, neeeds support for secondary video in detecting bad frames)




                
                    
                   
                    



    for index in range(number,length_frames_folder):

        #frames_folder_iter = iter(frames_folder)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                
                image = frames_folder[index]
                current_frames_path = os.path.join(frames_folder_path, image) 
                current_image = cv2.imread(current_frames_path)
                if index == length_frames_folder-1:
                    print('detecting bad frames done')
                    break
                next_frame = frames_folder[index+1]
                next_frames_path = os.path.join(frames_folder_path,next_frame)

                
                next_image = cv2.imread(next_frames_path)

                # Convert the BGR image to RGB before processing
                results_1 = pose.process(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))
                results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

                # Get most important landmarks
                nose_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_1.pose_landmarks else None
                nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None
                if landmark_positioning_nose_1[0,index] == 0:
                    landmark_positioning_nose_1[0,index] = nose_landmark_1.x
                    landmark_positioning_nose_1[1,index] = nose_landmark_1.y

                left_wrist_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_1.pose_landmarks else None
                left_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_2.pose_landmarks else None
                if landmark_positioning_left_wrist_1[0,index] == 0:
                    landmark_positioning_left_wrist_1[0,index] = left_wrist_landmark_1.x
                    landmark_positioning_left_wrist_1[1,index] = left_wrist_landmark_1.y

                right_wrist_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_1.pose_landmarks else None
                right_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_2.pose_landmarks else None
                if landmark_positioning_right_wrist_1[0,index] == 0:
                    landmark_positioning_right_wrist_1[0,index] = right_wrist_landmark_1.x
                    landmark_positioning_right_wrist_1[1,index] = right_wrist_landmark_1.y

                left_ankle_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_1.pose_landmarks else None
                left_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_2.pose_landmarks else None
                if landmark_positioning_left_ankle_1[0,index] == 0:
                    landmark_positioning_left_ankle_1[0,index] = left_ankle_landmark_1.x
                    landmark_positioning_left_ankle_1[1,index] = left_ankle_landmark_1.y

                right_ankle_landmark_1 = results_1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_1.pose_landmarks else None
                right_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_2.pose_landmarks else None
                if landmark_positioning_right_ankle_1[0,index] == 0:
                    landmark_positioning_right_ankle_1[0,index] = right_ankle_landmark_1.x
                    landmark_positioning_right_ankle_1[1,index] = right_ankle_landmark_1.y

                # Check if nose landmarks are detected in both images
                if nose_landmark_1 and nose_landmark_2:
                    dx = nose_landmark_1.x - nose_landmark_2.x
                    dy = nose_landmark_1.y - nose_landmark_2.y  
                    counter = 0
                    '''change (interpolate) coordinates nose_landmark_between.x or y to point between nose_landmark_1.x or y und nose_landmark_2.x or y with help of counter if more than one frame is bad'''
                    # Check if landmarks move unrealisticly much
                    while dx > 0.02 or dx < -0.02 or dy > 0.02 or dy < -0.02 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad nose pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            nose_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = nose_landmark_1.x - nose_landmark_2.x
                                dy = nose_landmark_1.y - nose_landmark_2.y
                            else:
                                print('No nose detected')
                        else:
                            print('Detecting bad nose frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_nose_1[0,index+j] = landmark_positioning_nose_1[0,index+j-1]+ddx
                                landmark_positioning_nose_1[1,index+j] = landmark_positioning_nose_1[1,index+j-1]+ddy
                                       
                else:
                    print('No nose detected')
                
                 # Check if left wrist landmarks are detected in both images
                if left_wrist_landmark_1 and left_wrist_landmark_2:
                    dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                    dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y  
                    counter = 0
                    
                    # Check if landmarks move unrealisticly much
                    while dx > 0.05 or dx < -0.05 or dy > 0.05 or dy < -0.05 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad left wrist pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            left_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                                dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y
                            else:
                                print('No left wrist detected')
                        else:
                            print('Detecting bad left wrist frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_left_wrist_1[0,index+j] = landmark_positioning_left_wrist_1[0,index+j-1]+ddx
                                landmark_positioning_left_wrist_1[1,index+j] = landmark_positioning_left_wrist_1[1,index+j-1]+ddy
                                       
                else:
                    print('No left wrist detected')


                 # Check if right wrist landmarks are detected in both images
                if right_wrist_landmark_1 and right_wrist_landmark_2:
                    dx = right_wrist_landmark_1.x - right_wrist_landmark_2.x
                    dy = right_wrist_landmark_1.y - right_wrist_landmark_2.y  
                    counter = 0
                    if right_wrist_landmark_1.x < highest_hand2 :
                        highest_hand2 = right_wrist_landmark_1.x
                        mark2 = index - number
                    # Check if landmarks move unrealisticly much
                    while dx > 0.05 or dx < -0.05 or dy > 0.05 or dy < -0.05 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad right wrist pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            right_wrist_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = left_wrist_landmark_1.x - left_wrist_landmark_2.x
                                dy = left_wrist_landmark_1.y - left_wrist_landmark_2.y
                            else:
                                print('No right wrist detected')
                        else:
                            print('Detecting bad right wrist frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_right_wrist_1[0,index+j] = landmark_positioning_right_wrist_1[0,index+j-1]+ddx
                                landmark_positioning_right_wrist_1[1,index+j] = landmark_positioning_right_wrist_1[1,index+j-1]+ddy
                                       
                else:
                    print('No right wrist detected')



                 # Check if left ankle landmarks are detected in both images
                if left_ankle_landmark_1 and left_ankle_landmark_2:
                    dx = left_ankle_landmark_1.x - left_ankle_landmark_2.x
                    dy = left_ankle_landmark_1.y - left_ankle_landmark_2.y  
                    counter = 0
                    
                    # Check if landmarks move unrealisticly much
                    while dx > 0.03 or dx < -0.03 or dy > 0.03 or dy < -0.03 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad left ankle pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            left_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = left_ankle_landmark_1.x - left_ankle_landmark_2.x
                                dy = left_ankle_landmark_1.y - left_ankle_landmark_2.y
                            else:
                                print('No left ankle detected')
                        else:
                            print('Detecting bad left ankle frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_left_ankle_1[0,index+j] = landmark_positioning_left_ankle_1[0,index+j-1]+ddx
                                landmark_positioning_left_ankle_1[1,index+j] = landmark_positioning_left_ankle_1[1,index+j-1]+ddy
                                       
                else:
                    print('No left ankle detected')  


                 # Check if right ankle landmarks are detected in both images
                if right_ankle_landmark_1 and right_ankle_landmark_2:
                    dx = right_ankle_landmark_1.x - right_ankle_landmark_2.x
                    dy = right_ankle_landmark_1.y - right_ankle_landmark_2.y  
                    counter = 0
                    
                    # Check if landmarks move unrealisticly much
                    while dx > 0.03 or dx < -0.03 or dy > 0.03 or dy < -0.03 :
                        counter = counter + 1
                        if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break
                        print('Bad right ankle pose detection in' + next_frames_path)
                        if index + counter < length_frames_folder:
                            next_frame = frames_folder[index + counter]
                            next_frames_path = os.path.join(frames_folder_path,next_frame)
                            next_image = cv2.imread(next_frames_path)
                            results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))
                            right_ankle_landmark_2 = results_2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE] if results_2.pose_landmarks else None
                            if nose_landmark_1 and nose_landmark_2:
                                dx = right_ankle_landmark_1.x - right_ankle_landmark_2.x
                                dy = right_ankle_landmark_1.y - right_ankle_landmark_2.y
                            else:
                                print('No right ankle detected')
                        else:
                            print('Detecting bad right ankle frames done')
                            break
                        
                    if counter  == 5:
                            print('Possibly the worst video quality ever. Try a new video.')
                            break 
                    else:
                        if counter != 0:
                            ddx = dx / (counter+1)
                            ddy = dy / (counter+1)
                        for j in range(1,counter+1):
                                landmark_positioning_right_ankle_1[0,index+j] = landmark_positioning_right_ankle_1[0,index+j-1]+ddx
                                landmark_positioning_right_ankle_1[1,index+j] = landmark_positioning_right_ankle_1[1,index+j-1]+ddy
                                       
                else:
                    print('No right ankle detected')                                        


    #cutting frames at the front so that the moment for synchronizing is at the same frame number (cutting frames isnt working yet)
    if mark1 > mark2:
        for i in range (0, mark1-mark2):
            #name of the frames from primary video
            file_path = './primary_frames_annotated/a_Federer_forehand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
            file_path = './frames/a_Federer_forehand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
            length_video1 = length_video1 - mark1 + mark2 
    else:
        for index in range (0, mark2-mark1):
            #name of the frames from secondary video
            file_path = './secondary_frames_annotated/Andriy_forehand_frame_' + str(index) + '.jpg'
            os.remove(file_path)
            file_path = './frames/Andriy_forehand_frame_' + str(index) + '.jpg'
            os.remove(file_path)
        #cut frames from video 2 at start amount of frames needed to be cut: mark2-mak1

        #updating video length
        length_video2 = length_video2 - mark2 + mark1


    #cutting frames at the end so that both video have same length
    if length_video1 > length_video2:
        for i in range (length_video2, length_video1):
            #name of the frames from primary video
            file_path = './primary_frames_annotated/a_Federer_forehand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
            file_path = './frames/a_Federer_forehand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
    else:
        #cut frames video2 at end with number higher length_video1
        for index in range (length_video1, length_video2):
            #name of the frames from secondary video
            
            file_path = './secondary_frames_annotated/Andriy_forehand_frame_' + str(index) + '.jpg'
            os.remove(file_path)
            file_path = './frames/Andriy_forehand_frame_' + str(index) + '.jpg'
            os.remove(file_path)


    # Folders containing relevant images
    #frames_folder_path_1 = 'primary_frames_annotated'
    #frames_folder_path_2 = 'secondary_frames_annotated'
    frames_folder_path = 'frames'

    # Get lists of image filenames from folder
    #frames_folder_1 = [filename for filename in os.listdir(frames_folder_path_1) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    #frames_folder_2 = [filename for filename in os.listdir(frames_folder_path_2) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    frames_folder = [filename for filename in os.listdir(frames_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    
    
    # Sort the frames after the numbers
    #primary_frames_folder = sorted(frames_folder_1)
    #secondary_frames_folder = sorted(frames_folder_2)
    frames_folder = sorted(frames_folder)

    length_frames_folder = len(frames_folder)        

    

    landmark_error = np.zeros(int(length_frames_folder/2),dtype=float)

    if mark1 > mark2:
        start1 = mark1-mark2
    else:
        start1 = 0
    start2 = length_video1 + abs(mark1-mark2)
        
    if length_video1 > length_video2:
        end1 = length_video2 + start1
        end2 = length_video1 + length_video2 + abs(mark1-mark2)
    else:
        end1 = length_video1 + start1
        end2 = length_video1 + length_video1 + abs(mark1-mark2)


    for j in range(0,int(length_frames_folder/2)):
        x_offset = landmark_positioning_nose_1[0,start1+j] - landmark_positioning_nose_1[0,start2+j]
        y_offset = landmark_positioning_nose_1[1,start1+j] - landmark_positioning_nose_1[1,start2+j]

        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:

                primary_frame = frames_folder[j]
                primary_frames_path = os.path.join(frames_folder_path, primary_frame) 
                primary_image = cv2.imread(primary_frames_path)
                results_1 = pose.process(cv2.cvtColor(primary_image, cv2.COLOR_BGR2RGB))

                secondary_frame = frames_folder[int(length_frames_folder/2)+j]
                secondary_frames_path = os.path.join(frames_folder_path,secondary_frame)
                secondary_image = cv2.imread(secondary_frames_path)
                results_2 = pose.process(cv2.cvtColor(secondary_image, cv2.COLOR_BGR2RGB))

                for (landmark1, landmark2) in zip(results_1.pose_landmarks.landmark, results_2.pose_landmarks.landmark):
                    x_error = landmark1.x - landmark2.x - x_offset
                    y_error = landmark1.y - landmark2.y - y_offset
                    landmark_error[j] = norm([x_error, y_error]) ** 2



    '''WiMa Code'''






    # Save errors in csv file
    import pandas as pd 
    
    error = []

    df = pd.DataFrame(error)
    df.to_csv("error.csv")               
                
    
    mp_pose = mp.solutions.pose

    # Folders containing relevant images
    frames_folder_path = 'frames'
    #images_folder_path = 'secondary_frames_annotated'

    # Get lists of image filenames from folder
    frames_folder = [filename for filename in os.listdir(frames_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    #images_folder = [filename for filename in os.listdir(images_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    
    
    
    # Sort the frames after the numbers
    frames_folder = sorted(frames_folder)
    #images_folder = sorted(images_folder, key=extract_number)

    # Divide frames 
    
    primary_frames_folder = frames_folder[0:int(length_frames_folder/2)]
    primary_frames_folder = sorted(primary_frames_folder, key=extract_number)
    secondary_frames_folder = frames_folder[int(length_frames_folder/2):length_frames_folder]
    secondary_frames_folder = sorted(secondary_frames_folder, key=extract_number)
    
    # Iterate over images pairwise from both folders
    #for idx, (image1, image2, image3) in enumerate(zip(primary_frames_folder, secondary_frames_folder, images_folder)):
    for idx, (image1, image2) in enumerate(zip(primary_frames_folder, secondary_frames_folder)):
        
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                primary_frames_path = os.path.join(frames_folder_path, image1) 
                secondary_frames_path = os.path.join(frames_folder_path, image2) 
                #image_path =  os.path.join(images_folder_path, image3)   
                primary_image = cv2.imread(primary_frames_path)
                secondary_image = cv2.imread(secondary_frames_path)
                #frame = cv2.imread(image_path)
                
                # Convert the BGR image to RGB before processing.
                results1 = pose.process(cv2.cvtColor(primary_image, cv2.COLOR_BGR2RGB))
                results2 = pose.process(cv2.cvtColor(secondary_image, cv2.COLOR_BGR2RGB))    
                
                # Normalize error values
                if landmark_error.max() > 0:
                    errors_normalized = (landmark_error - landmark_error.min()) / (landmark_error.max() - landmark_error.min())
                else:
                    errors_normalized = landmark_error  
                            
                # Create color gradient
                    
                def custom_colormap():
                    # Define colors
                    green = np.array([0, 255, 0])  # Green
                    yellow = np.array([0, 255, 255]) # Yellow
                    red = np.array([0, 0, 255])     # Red
                    
                    # Create colormap
                    cmap = np.zeros((256, 1, 3), dtype=np.uint8)
                    for i in range(256):
                        ratio = i / 255.0
                        if ratio < 0.5:
                            cmap[i, 0, :] = (1 - 2 * ratio) * green + (2 * ratio) * yellow
                        else:
                            cmap[i, 0, :] = (1 - ratio) * yellow + (2 * ratio - 1) * red
                    
                    return cmap
                
                colors = cv2.applyColorMap((errors_normalized * 255).astype(np.uint8), custom_colormap())
                
                for index, landmark2 in enumerate(results2.pose_landmarks.landmark):

                    height, width, _ = secondary_image.shape
                    #height, width, _ = frame.shape
                    cx, cy = int(landmark2.x * width), int(landmark2.y * height)

                    error = errors_normalized[index]

                    # Adjust color base on error values
                    color = tuple(map(int, colors[index][0]))
                    
                    # Draw landmarks
                    image = cv2.circle(secondary_image, (cx, cy), 3, color, -1)
                    #image = cv2.circle(frame, (cx, cy), 3, color, -1)

                #landmark_error = []    

        # Generate a unique filename for the combined image
        image_filename = f'error_landmarks_{idx}.jpg'  
                
        # Save the resulting image
        image_path = os.path.join('error_landmarks', image_filename)
        cv2.imwrite(image_path, image)       
                    
    
    # Draw landmarks on white background
    
    # import os
    # import cv2
    # import numpy as np
    # import mediapipe as mp
    # from mediapipe import solutions
    # from mediapipe.framework.formats import landmark_pb2  

    mp_pose = mp.solutions.pose

    # Folders containing relevant images
    frames_folder_path = 'frames'

    # Get lists of image filenames from both folders
    frames_folder = [filename for filename in os.listdir(frames_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Function for extracting the number from a frame
    def extract_number(frame):
       return int(''.join(filter(str.isdigit, frame)))
    
    # Sort the frames after the numbers
    frames_folder = sorted(frames_folder)

    # Divide frames 
    primary_frames_folder = frames_folder[0:int(length_frames_folder/2)]
    primary_frames_folder = sorted(primary_frames_folder, key=extract_number)
    secondary_frames_folder = frames_folder[int(length_frames_folder/2):length_frames_folder]
    secondary_frames_folder = sorted(secondary_frames_folder, key=extract_number)
    
    # Iterate over images pairwise from both folders
    for idx, (image1, image2) in enumerate(zip(primary_frames_folder, secondary_frames_folder)):
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                primary_frames_path = os.path.join(frames_folder_path, image1) 
                secondary_frames_path = os.path.join(frames_folder_path, image2)   
                primary_image = cv2.imread(primary_frames_path)
                secondary_image = cv2.imread(secondary_frames_path)
                # Convert the BGR image to RGB before processing.
                results1 = pose.process(cv2.cvtColor(primary_image, cv2.COLOR_BGR2RGB))
                results2 = pose.process(cv2.cvtColor(secondary_image, cv2.COLOR_BGR2RGB))
            
                
        # Create white frame as canvas for the landmarks
        
        # Get the height and width of the frames
        height1, width1 = primary_image.shape[:2]
        height2, width2 = secondary_image.shape[:2]
        primary_frame = np.full([height1, width1, 3], [255, 255, 255] , dtype=np.uint8)    
        secondary_frame = np.full([height2, width2, 3], [255, 255, 255], dtype=np.uint8)               
                
        if results1.pose_landmarks is not None:
            for landmark in results1.pose_landmarks.landmark:
                #height, width, _ = frame.shape
                cx, cy = int(landmark.x * width1), int(landmark.y * height1)
                primary_image = cv2.circle(primary_frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw a red circle at each landmark
        
        if results2.pose_landmarks is not None:
            for landmark in results2.pose_landmarks.landmark:
                #height, width, _ = frame.shape
                cx, cy = int(landmark.x * width2), int(landmark.y * height2)
                secondary_image = cv2.circle(secondary_frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw a red circle at each landmark
     

        # Generate a unique filename for the image
        primary_image_filename = f'primary_landmarks_{idx}.jpg'  
        secondary_image_filename = f'secondary_landmarks_{idx}.jpg'  

        # Save the resulting image
        primary_image_path = os.path.join('frames_landmarks', primary_image_filename)
        secondary_image_path = os.path.join('frames_landmarks', secondary_image_filename)
        cv2.imwrite(primary_image_path, primary_image)  
        cv2.imwrite(secondary_image_path, secondary_image)        




    '''delete from here?'''





    # Body Segmentation

    import os
    import cv2
    import numpy as np
    import torch
    
    # Load trained model
    
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()

    from PIL import Image
    from torchvision import transforms

    folder_path = 'frames'

    # Get lists of image filenames from folder
    folder_images = [filename for filename in os.listdir(folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    
    # Function for extracting the number from a frame
    def extract_number(frame):
       return int(''.join(filter(str.isdigit, frame)))
    
    # Sort the frames after the numbers
    folder_images = sorted(folder_images)

    # Get relevant frames from secondary video
    folder_images = folder_images[int(length_frames_folder/2):int(length_frames_folder)]
    folder_images = sorted(folder_images, key=extract_number)
    
    # Iterate over images from folder
    for idx, image in enumerate(folder_images):
        path = os.path.join(folder_path, image)
        input_image = Image.open(path)
        #input_image = Image.open('secondary_frames_annotated/Teana_forhand_frame_0.jpg')
        input_image = input_image.convert("RGB")
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # Define the color for segmentation
        white_color = torch.tensor([255, 255, 255])
        gray_color = torch.tensor([192, 192, 192])

        # Create a color palette for each class
        palette = torch.tensor([[2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1] for _ in range(21)])
    
        # Replace the first color (background color) with white
        palette[0] = white_color
        palette[1:] = gray_color
    
        # Generate colors for each class using the modified palette
        colors = (palette).numpy().astype("uint8")

        r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(input_image.size)
        r.putpalette(colors)

        # Convert the image to RGB mode before saving as JPG
        r = r.convert('RGB')

        # Generate a unique filename for the segmented frame
        segmented_image_filename = f'segmented_frame_{idx}.jpg'  

        # Save the resulting segmented image
        segmented_image_path = os.path.join('segmented_frames', segmented_image_filename)
        r.save(segmented_image_path)
        

    # Draw landmarks on the segmented frames 
    
    # import os
    # import cv2
    # import mediapipe as mp
    # from mediapipe import solutions
    # from mediapipe.framework.formats import landmark_pb2  

    mp_pose = mp.solutions.pose

    # Folders containing relevant images
    frames_folder_path = 'frames'
    segmented_folder_path = 'segmented_frames'

    # Get lists of image filenames from both folders
    frames_folder = [filename for filename in os.listdir(frames_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    segmented_folder = [filename for filename in os.listdir(segmented_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    # Function for extracting the number from a frame
    def extract_number(frame):
       return int(''.join(filter(str.isdigit, frame)))
    
    # Sort the frames after the numbers
    frames_folder = sorted(frames_folder)
    segmented_folder = sorted(segmented_folder, key=extract_number)

    # Get relevant frames from secondary video
    #start_index = 65
    #frames_folder = frames_folder[start_index:]
    frames_folder = frames_folder[0:int(length_frames_folder)]
    frames_folder = sorted(frames_folder, key=extract_number)
    
    # Iterate over images pairwise from both folders
    for idx, (image1, image2) in enumerate(zip(frames_folder, segmented_folder)):
        
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                frames_path = os.path.join(frames_folder_path, image1)   
                image = cv2.imread(frames_path)
                # Convert the BGR image to RGB before processing.
                results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            
        segmented_path = os.path.join(segmented_folder_path, image2)    
        frame = cv2.imread(segmented_path) 

        if results.pose_landmarks is not None:
            for landmark in results.pose_landmarks.landmark:
                height, width, _ = frame.shape
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                image = cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)  # Draw a red circle at each landmark

        # Generate a unique filename for the combined image
        image_filename = f'segmented_frame_annotated_{idx}.jpg'  

        # Save the resulting image
        image_path = os.path.join('segmented_frames_annotated', image_filename)
        cv2.imwrite(image_path, image)        
    

    # Overlay images for comparison

    # import cv2
    # import numpy as np
    
    # MediaPipe Pose Estimation initialization
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=True, 
                        model_complexity=2, 
                        enable_segmentation=True,
                        min_detection_confidence=0.5)

    # Extract landmarks
    landmark_image1 = cv2.imread('frames/a_Federer_forehand_frame_0.jpg')
    landmark_image2 = cv2.imread('frames/Andriy_forehand_frame_0.jpg')

    # Adjust image shape for comparison
    landmark_image2 = cv2.resize(landmark_image2,(1920, 1080)) # Original shape: (848, 480, 3)

    # Perform pose estimation on the first image and find the landmarks
    results1 = pose.process(cv2.cvtColor(landmark_image1, cv2.COLOR_BGR2RGB))
    nose_landmark_1 = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results1.pose_landmarks else None
    left_heel_landmark1 = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL] if results1.pose_landmarks else None
    right_heel_landmark1 = results1.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL] if results1.pose_landmarks else None

    # Perform pose estimation on the second image and find the landmarks
    results2 = pose.process(cv2.cvtColor(landmark_image2, cv2.COLOR_BGR2RGB))
    nose_landmark_2 = results2.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE] if results2.pose_landmarks else None
    left_heel_landmark2 = results2.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HEEL] if results1.pose_landmarks else None
    right_heel_landmark2 = results2.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HEEL] if results1.pose_landmarks else None

    # Folders containing relevant images
    primary_frames_folder_path = 'primary_frames_annotated'
    segmented_folder_path = 'segmented_frames_annotated'
    #segmented_folder_path = 'secondary_frames_annotated'

    # Get lists of image filenames from both folders
    primary_frames_folder = [filename for filename in os.listdir(primary_frames_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]
    segmented_folder = [filename for filename in os.listdir(segmented_folder_path) if filename.endswith(('.jpg', '.png', '.jpeg'))]

    # Sort the frames after the numbers
    primary_frames_folder = sorted(primary_frames_folder, key=extract_number)
    segmented_folder = sorted(segmented_folder, key=extract_number)



    #delete untill here?




    # Iterate over images pairwise from both folders
    for idx, (image1, image2) in enumerate(zip(primary_frames_folder, segmented_folder)):
        
        primary_frames_path = os.path.join(primary_frames_folder_path, image1)   
        image1 = cv2.imread(primary_frames_path, cv2.IMREAD_UNCHANGED)
        image1 = cv2.cvtColor(image1,cv2.COLOR_BGR2BGRA)               # Add alpha channel
        segmented_path = os.path.join(segmented_folder_path, image2)   
        image2 = cv2.imread(segmented_path, cv2.IMREAD_UNCHANGED)
        image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2BGRA)               # Add alpha channel

        # Resize image for aligning the body
        image2 = cv2.resize(image2,(1920, 1080)) # Original shape: (848, 480, 3)
            
        
        # Compute relevant region around both bodies for overlaying

        nose_y_1 = int(nose_landmark_1.y * image1.shape[0])        # reference point for the top edge of the first region
        nose_y_2 = int(nose_landmark_2.y * image2.shape[0])        # reference point for the top edge of the second region

        l_heel_x_1 = int(left_heel_landmark1.x * image1.shape[1])  # reference point for the left edge of the first region
        l_heel_x_2 = int(left_heel_landmark2.x * image2.shape[1])  # reference point for the left edge of the second region
        l_heel_y_1 = int(left_heel_landmark1.y * image1.shape[0])  # reference point for the bottom edge of the first region
        l_heel_y_2 = int(left_heel_landmark2.y * image2.shape[0])  # reference point for the bottom edge of the second region
        
        r_heel_x_1 = int(right_heel_landmark1.x * image1.shape[1]) # reference point for the right edge of the first region
        r_heel_x_2 = int(right_heel_landmark2.x * image2.shape[1]) # reference point for the right edge of the second region
        r_heel_y_1 = int(right_heel_landmark1.y * image1.shape[0]) # reference point for the bottom edge of the first region
        r_heel_y_2 = int(right_heel_landmark2.y * image2.shape[0]) # reference point for the bottom edge of the second region
        
        # Adjust the reference points to get the regions surrounding the corresponding body in all frames
        
        right_x_1 = r_heel_x_1 - 120                         
        left_x_1 = l_heel_x_1 + 50
        bottom_y_1 = max(r_heel_y_1 + 30 ,l_heel_y_1 + 30)
        top_y_1 = nose_y_1 - 120

        right_x_2 = r_heel_x_2 - 120
        left_x_2 = l_heel_x_2 + 35
        bottom_y_2 = max(r_heel_y_2 + 30 ,l_heel_y_2 + 30)
        top_y_2 = nose_y_2 - 120
        
        diff_x = left_x_1-right_x_1 - (left_x_2-right_x_2) 
        diff_y = bottom_y_1 - top_y_1 - (bottom_y_2-top_y_2)
        
        alpha = 0.5  # Factor for transparency

        # Distribute the differences in width and height of both regions evenly

        if diff_x < 0:
            diff_x = - diff_x
            rest_x = -(diff_x % 4)
        else:
            rest_x = diff_x % 4
        
        if diff_y < 0:
            diff_y = - diff_y
            rest_y = -(diff_y % 4)    
        else:
            rest_y = diff_y % 4

        if (diff_x % 4 != 0) and (diff_y % 4 != 0):
            image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] = np.uint8(
                image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] * alpha +  
                image2[top_y_2-int(diff_y/4)-rest_y:bottom_y_2+int(diff_y/4), right_x_2-int(diff_x/4)-rest_x:left_x_2+int(diff_x/4), :]   * (1-alpha)
            )
        elif (diff_x % 4) != 0:
            image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] = np.uint8(
                image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] * alpha +  
                image2[top_y_2-int(diff_y/4):bottom_y_2+int(diff_y/4), right_x_2-int(diff_x/4)-rest_x:left_x_2+int(diff_x/4), :]   * (1-alpha)
            )  
        elif (diff_y % 4) != 0:
            image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] = np.uint8(
                image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] * alpha +  
                image2[top_y_2-int(diff_y/4)-rest_y:bottom_y_2+int(diff_y/4), right_x_2-int(diff_x/4):left_x_2+int(diff_x/4), :]   * (1-alpha)
            )         
        else:
            image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] = np.uint8(
                image1[top_y_1+int(diff_y/4):bottom_y_1-int(diff_y/4), right_x_1+int(diff_x/4):left_x_1-int(diff_x/4), :] * alpha +  
                image2[top_y_2-int(diff_y/4):bottom_y_2+int(diff_y/4), right_x_2-int(diff_x/4):left_x_2+int(diff_x/4), :]   * (1-alpha)
            )     
        

        # Generate a unique filename for the combined image
        combined_image_filename = f'combined_image_{idx}.jpg'  

        # Save the resulting combined image 
        combined_image_path = os.path.join('frames_annotated', combined_image_filename)
        cv2.imwrite(combined_image_path, image1)



    # created annotated movie from annotated frames
    
    import cv2
    import os

    # Directory containing frames
    primary_frames_dir = 'primary_frames_annotated'
    secondary_frames_dir = 'secondary_frames_annotated'
    comparison_frames_dir = 'frames_annotated'

    # Output video file name
    primary_video_filename = 'primary_video.mp4'
    secondary_video_filename = 'secondary_video.mp4'
    comparison_video_filename = 'comparison_video.mp4'

    # Get the list of frames in the directory
    primary_frames = [f for f in os.listdir(primary_frames_dir) if f.endswith('.jpg')]
    secondary_frames = [f for f in os.listdir(secondary_frames_dir) if f.endswith('.jpg')]
    comparison_frames = [f for f in os.listdir(comparison_frames_dir) if f.endswith('.jpg')]

    # Sort the frames to ensure correct order

    # Function for extracting the number from a frame
    def extract_number(frame):
        return int(''.join(filter(str.isdigit, frame)))

    # Sort the frames after the numbers
    primary_frames = sorted(primary_frames, key=extract_number)
    secondary_frames = sorted(secondary_frames, key=extract_number)
    comparison_frames = sorted(comparison_frames, key=extract_number)
    
    # Get the first frame to obtain frame size information
    primary_frame_path = os.path.join(primary_frames_dir, primary_frames[0])
    primary_frame = cv2.imread(primary_frame_path)
    primary_height, primary_width, primary_layers = primary_frame.shape

    secondary_frame_path = os.path.join(secondary_frames_dir, secondary_frames[0])
    secondary_frame = cv2.imread(secondary_frame_path)
    secondary_height, secondary_width, secondary_layers = secondary_frame.shape

    comparison_frame_path = os.path.join(comparison_frames_dir, comparison_frames[0])
    comparison_frame = cv2.imread(comparison_frame_path)
    comparison_height, comparison_width, comparison_layers = comparison_frame.shape

    # Define the codec and create a VideoWriter object
    video_path = 'videos_annotated'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    primary_video = cv2.VideoWriter(video_path+'/primary_video.mp4', fourcc, 30.0, (primary_width, primary_height))
    secondary_video = cv2.VideoWriter(video_path+'/secondary_video.mp4', fourcc, 30.0, (secondary_width, secondary_height))
    comparison_video = cv2.VideoWriter(video_path+'/comparison_video.mp4', fourcc, 5.0, (comparison_width, comparison_height))

    # Loop through the frames and add them to the video
    for frame_name in primary_frames:
        primary_frame_path = os.path.join(primary_frames_dir, frame_name)
        primary_frame = cv2.imread(primary_frame_path)
        primary_video.write(primary_frame)

    for frame_name in secondary_frames:
        secondary_frame_path = os.path.join(secondary_frames_dir, frame_name)
        secondary_frame = cv2.imread(secondary_frame_path)
        secondary_video.write(secondary_frame)    
    
    for frame_name in comparison_frames:
        comparison_frame_path = os.path.join(comparison_frames_dir, frame_name)
        comparison_frame = cv2.imread(comparison_frame_path)
        comparison_video.write(comparison_frame)

    # Release the VideoWriter and close all OpenCV windows
    primary_video.release()
    secondary_video.release()
    comparison_video.release()
    cv2.destroyAllWindows()

    print(f'Video {primary_video_filename} created successfully.')
    print(f'Video {secondary_video_filename} created successfully.')
    print(f'Video {comparison_video_filename} created successfully.')





# Check if this script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()  # Call the main function


