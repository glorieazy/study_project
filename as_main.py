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
                                name="nadal_serve.mp4") 
    manager.secondary = asVideo( dir=manager.config.get_value("dirs","media"),  
                                  name="jonas_serve.mp4") 
    
    
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
    '''half_frames = len(frames_folder)/2

    primary_frames_folder = frames_folder[half_frames:]
    primary_frames_folder = sorted(primary_frames_folder, key=extract_number)
    secondary_frames_folder = frames_folder[:half_frames]
    secondary_frames_folder = sorted(secondary_frames_folder, key=extract_number)

    frames_folder_iter = iter(frames_folder)'''

    length_frames_folder = len(frames_folder)

    landmark_positioning_nose_1 = np.zeros((2,length_frames_folder),dtype = float)
    landmark_positioning_nose_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_left_wrist_1 = np.zeros((2,length_frames_folder),dtype = float)
    landmark_positioning_left_wrist_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_right_wrist_1 = np.zeros((2,length_frames_folder),dtype = float)
    landmark_positioning_right_wrist_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_left_ankle_1 = np.zeros((2,length_frames_folder),dtype = float)
    landmark_positioning_left_ankle_2 = np.zeros((2,length_frames_folder/2),dtype = float)

    landmark_positioning_right_ankle_1 = np.zeros((2,length_frames_folder),dtype = float)
    landmark_positioning_right_ankle_2 = np.zeros((2,length_frames_folder/2),dtype = float)
    mark1=0 
    highest_hand1=0
    mark2=0
    highest_hand2=0
    # Go through frames of one video and put landmark positions in an array
    for index in range(0,length_frames_folder/2):

        #frames_folder_iter = iter(frames_folder)

        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.5) as pose:
                
                image = frames_folder[index]
                current_frames_path = os.path.join(frames_folder_path, image) 
                current_image = cv2.imread(current_frames_path)
                if index == length_frames_folder/2-1:
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
                    while dx > 0.02 | dx < -0.02 | dy > 0.02 | dy < -0.02 :
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
                    while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
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
                    if right_wrist_landmark_1.x > highest_hand1 :
                        highest_hand1 = right_wrist_landmark_1
                        mark1 = index
                    # Check if landmarks move unrealisticly much
                    while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
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
                    while dx > 0.03 | dx < -0.03 | dy > 0.03 | dy < -0.03 :
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
                    while dx > 0.03 | dx < -0.03 | dy > 0.03 | dy < -0.03 :
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




                
                    
                   
                    



    for index in range(length_frames_folder/2,length_frames_folder):

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
                    while dx > 0.02 | dx < -0.02 | dy > 0.02 | dy < -0.02 :
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
                    while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
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
                    if right_wrist_landmark_1.x > highest_hand2 :
                        highest_hand2 = right_wrist_landmark_1
                        mark2 = index
                    # Check if landmarks move unrealisticly much
                    while dx > 0.05 | dx < -0.05 | dy > 0.05 | dy < -0.05 :
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
                    while dx > 0.03 | dx < -0.03 | dy > 0.03 | dy < -0.03 :
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
                    while dx > 0.03 | dx < -0.03 | dy > 0.03 | dy < -0.03 :
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



    landmark_error = np.zeros((length_frames_folder/2))

    for j in range(0,length_frames_folder/2):
        x_offset = landmark_positioning_nose_1[0,j] - landmark_positioning_nose_1[0,length_frames_folder/2+1]
        y_offset = landmark_positioning_nose_1[1,j] - landmark_positioning_nose_1[1,length_frames_folder/2+1]

        image = frames_folder[j]
        current_frames_path = os.path.join(frames_folder_path, image) 
        current_image = cv2.imread(current_frames_path)
        results_1 = pose.process(cv2.cvtColor(current_image, cv2.COLOR_BGR2RGB))

        next_frame = frames_folder[length_frames_folder/2 + j]
        next_frames_path = os.path.join(frames_folder_path,next_frame)
        next_image = cv2.imread(next_frames_path)
        results_2 = pose.process(cv2.cvtColor(next_image, cv2.COLOR_BGR2RGB))

        for (landmark1, landmark2) in zip(results_1.pose_landmarks.landmark, results_2.pose_landmarks.landmark):
            x_error = landmark1.x - landmark2.x - x_offset
            y_error = landmark1.y - landmark2.y - y_offset
            landmark_error[j] = norm([x_error, y_error]) ** 2


    #cutting frames at the front so that the moment for synchronizing is at the same frame number (cutting frames isnt working yet)
    if mark1 > mark2:
        for i in range (0, mark1-mark2):
            #name of the frames from primary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
            length_video1 = length_video1 - mark1 +mark2 
    else:
        for index in range (0, mark2-mark1):
            #name of the frames from secindary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
        #cut frames from video 2 at start amount of frames needed to be cut: mark2-mak1

        #updating video length
        length_video2 = length_video2 - mark2 + mark1


    #cutting frames at the end so that both video have same length
    if length_video1 > length_video2:
        for i in range (length_video2, length_video1):
            #name of the frames from primary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
    else:
        #cut frames video2 at end with number higher length_video1
        for i in range (length_video1, length_video2):
            #name of the frames from secondary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)







'''

    # frame number and height for highest hand point for time synchronizing
    mark1=0 
    highest_hand1=0
    mark2=0
    highest_hand2=0


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
                    
                    #getting amount of frames in video
                    length_video1=index

                    
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
                            
                            else:
                                #getting height for the right hand primary video change x for y depending if serve or forehand
                                if right_wrist_landmark_1.x > highest_hand1 :
                                    highest_hand1 = right_wrist_landmark_1
                                    mark1 = index

                                #getting height for the right hand secondary video
                            

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








    #time synchronizing (code here isnt working yet, neeeds support for secondary video in detecting bad frames)

    #cutting frames at the front so that the moment for synchronizing is at the same frame number (cutting frames isnt working yet)
    if mark1 > mark2:
        for i in range (0, mark1-mark2):
            #name of the frames from primary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
            length_video1 = length_video1 - mark1 +mark2 
    else:
        for index in range (0, mark2-mark1):
            #name of the frames from secindary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
        #cut frames from video 2 at start amount of frames needed to be cut: mark2-mak1

        #updating video length
        length_video2 = length_video2 - mark2 + mark1


    #cutting frames at the end so that both video have same length
    if length_video1 > length_video2:
        for i in range (length_video2, length_video1):
            #name of the frames from primary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)
    else:
        #cut frames video2 at end with number higher length_video1
        for i in range (length_video1, length_video2):
            #name of the frames from secondary video
            file_path = './frames_annotated/Daria_forhand_frame_' + str(i) + '.jpg'
            os.remove(file_path)

    











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