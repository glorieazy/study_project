from as_ImageProcessor import asImageProcessor


class asFrame():
    """
    This class contains information about captured frame
    """
    def __init__(self, name, time, frame, video):        
        self.name = name
        self.video = video # Super video
        self.time = time # Time frame in video
        
        # inner parameters of the frame
        self.frame = frame # Image array
        self.width = video.width
        self.height = video.height

        self.detection_result = None
        self.NormalizedLandmarks = []
        self.Landmarks = []
        self.segmentation_mask = []

        # status
        self.is_processed = False


    def resize(self, shape, to_change_file=False):
        """
        resize frame
        Input:
        shape(width, height)
        """
        from as_Video import asVideo        
        import cv2 
        
        resized =cv2.resize( self.frame, shape, 
                             interpolation=cv2.INTER_NEAREST)
        self.frame = resized
        #
        # change height and width of the image
        self.width  = shape[0]
        self.height = shape[1]

        if to_change_file:
            self.write_frame(self.name, self.frame)


    def show(self, text=""):
        import cv2 
        cv2.imshow(text, self.frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def display_info(self, is_detailed=False):
        """
        Dispays the information about frame
        """
        print()        
        print("----- Information about frame")        
        print(f"Frame name: {self.name}")
        print(f"Stamming from video: {self.video.name}")
        print(f"at time: {self.time}")
        print(f"It is processed: {self.is_processed}")
        #
        if is_detailed:
            print("-----------the detailed infromation-------------")
            print(f"... to be added")


    ##########################
    ### processing methods ###
    ##########################
    def process_frame(self, image_processor : asImageProcessor):
        """
        process the curent frame
        Input: 
            image_processor
        Output:
            detection_result            
        """
        #print( self.name )
        self.detection_result = image_processor.process_frame( self.name )


    def show_annotated_frame(self, image_processor : asImageProcessor):
        """
        show_annotated_frame
        """
        import mediapipe as mp

        if self.detection_result is None:
            self.process_frame(image_processor)

        image = mp.Image.create_from_file( self.name )
        # create annotatated image
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), self.detection_result)
        # show it
        asFrame.show_frame( annotated_image, text="Annotated image for " + self.name )


    def save_annotated_frame(self, image_processor : asImageProcessor, dir_to_save):
        """
        save_annotated_frame
        """
        import mediapipe as mp

        if self.detection_result is None:
            self.process_frame(image_processor)

        image = mp.Image.create_from_file( self.name )
        # create annotatated image
        annotated_image = self.draw_landmarks_on_image(image.numpy_view(), self.detection_result)
        # save it
        # words = self.name.split("/") 
        #print( self.name.split("/") )
        path_to_save = "./" + dir_to_save + "/" + self.name.split("/")[-1]
        #print( path_to_save )
        asFrame.write_frame(path_to_save, annotated_image)


    def show_landmarks_frame(self, image_processor : asImageProcessor):
        """
        show_landmarks_frame
        """
        pass


    def save_landmarks_frame(self, image_processor : asImageProcessor, path_to_save):
        """
        save_landmarks_frame
        """
        pass


    def draw_landmarks_on_image(self, rgb_image, detection_result):    
        from mediapipe import solutions
        from mediapipe.framework.formats import landmark_pb2
        import numpy as np 

        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                pose_landmarks_proto,
                solutions.pose.POSE_CONNECTIONS,
                solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image


    def show_segmentation_mask(self):
        """
        show the segmentated image
        """
        import cv2
        import numpy as np

        #from as_ImageProcessor import asImageProcessor
        if self.detection_result is None:
            raise Exception(f"Exception: no detection results for the frame {self.name}")

        segmentation_mask = self.detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        #asFrame.show_frame( annotated_frame, text=f"Annotated frame {filename}")
        cv2.imshow('Image Window', visualized_mask)
        #cv2_imshow(visualized_mask)


    def get_segmentation_mask(self):
        """
        returns the segmentation-mask (white spot) of the particular-self frame
        Output:
            visualized_mask
        """
        import cv2
        import numpy as np

        #from as_ImageProcessor import asImageProcessor
        if self.detection_result is None:
            raise Exception(f"Exception: no detection results for the frame {self.name}")

        segmentation_mask = self.detection_result.segmentation_masks[0].numpy_view()
        visualized_mask = np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
        #asFrame.show_frame( annotated_frame, text=f"Annotated frame {filename}")
        #cv2.imshow('Image Window', visualized_mask)
        #cv2_imshow(visualized_mask)
        return visualized_mask
        

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


    @staticmethod
    def show_frame( frame, text=""):
        import cv2 
        cv2.namedWindow(text)
        cv2.moveWindow(text, 10, 10)
        cv2.imshow(text, frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()