###################################################
### This class manageres all videos in this app ###
###################################################

class asVideosManager:

    def __init__(self):
        # attrs
        self.config          = None
        self.primary         = None
        self.secondary       = None
        self.secondaries     = None
        self.logger          = None 
        self.image_processor = None

    def add_config( self, config ):
        """
        Add config to manager
        """
        self.config = config

    def add_primary( self, primary ):
        """
        Add primary video to manager
        """
        self.primary = primary

    def add_secondary( self, secondary ):
        """
        Add secondary video to manager
        """
        self.secondary = secondary

    def add_logger( self, logger ):
        """
        Add logger to manager
        """
        self.logger = logger

    def add_image_processor( self, image_processor ):
        """
        Add image_processor to manager
        """
        self.image_processor = image_processor

    ###################################
    ### decompose video into frames ###
    ###################################
    def decompose_into_frames(self, video=None, append_frames=True ):
        """        
        decompose video into frames and save them into 
        the special folder  config.get_value("dirs","frames")
        """
        import numpy as np
        from as_Video import asVideo        
        from as_Frame import asFrame        
        import cv2 

        if video is None:
            raise Exception("Exception: There is no video specified in manager.decompose_into_frames() ...")

        if append_frames:
            from as_Frame import asFrame

        cap = cv2.VideoCapture( video.get_fullname() )
        counter = 0
        while True:
            # Extract frames
            ret, frame = cap.read()
            if not ret:
                break

            # name = path_to_deconstruct + 'frame' + str(counter) + '.jpg'
            # print ('Creatings...' + name)
            # writing the extracted images
            name = self.get_full_framename(video, counter)            
            #print(f"Writing { name } ...")
            #print( frame[0,0,:] )
            #cv2.imwrite(name, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])            
            asFrame.write_frame(name, frame)
            #cv2.imwrite(name, frame)
            #image = cv2.imread(name, cv2.IMREAD_UNCHANGED)            
            #print( image[0,0,:] )
            #print()
            #print(frame.shape)
            #print(type(frame))
            # or

            if append_frames:                
                video.frames.append(asFrame(name, counter, np.array(frame), video))
                #video.frames.append(asFrame(name, counter, frame, video))
                #print( type(frame) )
                #print( frame.shape )
                #print()
            counter  = counter + 1


            # cv2.imshow("Original", frame)
            # cv2.waitKey(0)
            # cv.destroyAllWindows()
            # resized =cv.resize(frame, shape, interpolation=cv.INTER_NEAREST)
            # cv.imshow("Resized", resized)
            #print() 
            # break           

        #     width = VIDEO_FILE.get(cv.CAP_PROP_FRAME_WIDTH )
        #     height = VIDEO_FILE.get(cv.CAP_PROP_FRAME_HEIGHT )
        #     fps =  VIDEO_FILE.get(cv.CAP_PROP_FPS)

        #     # write down an image
        #     cv.imwrite(name, frame)

            # print(np.array(frame))
            #cv2.resize(frame, shape, interpolation=cv.INTER_NEAREST)
            #self.Captures.append(Capture(counter, np.array(frame), self))
            #time += 1

        # release the device
        cap.release()


    def get_framename(self, video, counter):
        """
        Constructs and returns framename: name + counter
        """
        name_without_extension = video.name.split(".")[0] 
        return name_without_extension + "_frame_" + str(counter) + '.jpg'


    def get_full_framename(self, video, counter):
        """
        Constructs and returns full_framename: dir + name + counter
        """
        return "./"  + self.config.get_value("dirs","frames") + "/" \
                        + self.get_framename(video, counter)


    ######################
    ### Static methods ###
    ######################
    @staticmethod
    def dump(manager):
        """
        dump manager
        """
        import pickle

        # Serialize (dump) the object to a file
        with open( manager.config.get_value("dump_data","dir") + '/' +
                   manager.config.get_value("dump_data","file") +'.pkl', 'wb') as file:
            pickle.dump(manager, file)


    @staticmethod
    def load():
        """
        load (create) manager from dump
        """
        import pickle

        # start config
        from as_config import asConfig
        config = asConfig("config/config.json")

        # Serialize (dump) the object to a file
        with open( config.get_value("dump_data","dir") + '/' +
                   config.get_value("dump_data","file") +'.pkl', 'rb') as file:
            manager = pickle.load(file)

        return manager
    

    ##############################################################
    ### methods, which are still not included into the diagram ###
    ##############################################################          
    def set_video(self, type_of_video="primary"):
        """
        Read video from config
        Input:
           type_of_video - "primary", secondary, ...
        """    
        from as_Video import asVideo
        dir = self.config.get_value("dirs","media") 
        name = self.config.get_value("videos",type_of_video) 

        # read video
        if type_of_video == "primary":
            self.primary = asVideo( dir=dir, name=name ) 
        elif type_of_video == "secondary":               
            self.secondary = asVideo( dir=dir, name=name )      
        else: 
            raise Exception("Incorrect type_of_video in <set_video>")            
        

    def init_param_video(self, type_of_video="primary"):
        """
        Initiliase parameters of video
        Set parameters        
        Input:
           type_of_video - "primary", secondary, ...
        """    
  
        # init params of video
        if type_of_video == "primary":
            self.primary.init_video()
            # self.primary.display_info(is_detailed=True)
        elif type_of_video == "secondary":               
            self.secondary.init_video()
            # self.secondary.display_info(is_detailed=True)
        else: 
            raise Exception("Incorrect type_of_video in <init_param_video>")            
    
    
    ##############################################################
    ### methods, which are still not included into the diagram ###
    ##############################################################          
    def decompose_video_into_frames(self, type_of_video="primary"):
        """
        Decompose primary and secondary videos into frames
        """
        # init params of video
        if type_of_video == "primary":
            self.decompose_into_frames( video=self.primary, append_frames=True )        
        elif type_of_video == "secondary":               
            self.decompose_into_frames( video=self.secondary, append_frames=True )        
        else: 
            raise Exception("Incorrect type_of_video in <decompose_video_into_frames>")            



                