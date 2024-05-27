class asImageProcessor():
    """
    This class contains image processor stuff:
    https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
    """    
    def __init__(self):
        self.detector = None

    def get_detector(self):
        """
        returns detector
        """
        return self.detector


    def initialize_detector(self):
        """
        Configures the pose detection in images:
        pose_landmarker.task is taken from 
        https://developers.google.com/mediapipe/solutions/vision/pose_landmarker#get_started
        (Pose landmarker (Heavy))
        """
        from mediapipe.tasks import python  
        from mediapipe.tasks.python import vision
        base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
        options = python.vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=True,
            min_pose_presence_confidence=0.5,
            min_pose_detection_confidence=0.5,
            num_poses=1, # as
            min_tracking_confidence=0.5)

        self.detector = vision.PoseLandmarker.create_from_options(options)
        #return detector


    def process_frame(self, image_path):
        """
        Process a single image/frame
        """
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        import cv2 

        if self.detector is None:
            raise Exception("Exception: You forgot to initialise detector by starting initialize_detector() ... ")

        # STEP 3: Load the input image.
        image = mp.Image.create_from_file( image_path )
        #print(type(image))
        #print()

        # STEP 4: Detect pose landmarks from the input image.
        detection_result = self.detector.detect(image)

        # if to_show:
        #     # STEP 5: Process the detection result. In this case, visualize it.
        #     annotated_image = self.draw_landmarks_on_image(image.numpy_view(), detection_result)

        #     from as_Frame import asFrame
        #     asFrame.show_frame( annotated_image, text="Annotated image for " + image_path )
        #     # #cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        #     # cv2.imshow('Image Window', annotated_image)

        #     # # Wait for a key press and then close the window
        #     # cv2.waitKey(0)
        #     # cv2.destroyAllWindows()

        return detection_result    


    #######################
    ### stattic methods ###
    #######################
    @staticmethod
    def read_frame(path):
        """
        read the frame
        Input:
        path - path from there to read
        """
        import cv2 
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)


    @staticmethod
    def write_frame(path, frame):
        """
        write the frame
        Input:
        path - path from there to read
        """
        import cv2 
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])            

