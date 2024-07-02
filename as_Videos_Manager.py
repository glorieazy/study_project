class asVideosManager:

    def __init__(self, config):
        self.config = config

        self._primary     = None
        self._secondary   = None
        self._secondaries = None

    @property
    def primary(self):
        return self._primary

    @primary.setter
    def primary(self, value):
        self._primary = value

    @property
    def secondary(self):
        return self._secondary

    @secondary.setter
    def secondary(self, value):
        self._secondary = value

    @property
    def secondaries(self):
        return self._secondaries

    @secondaries.setter
    def secondaries(self, videos):
        self._secondaries = []
        for item in videos:
            self._secondaries.append(item)

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