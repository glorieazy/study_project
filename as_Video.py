
from typing import List
from as_Frame import asFrame

class asVideo:

    def __init__(self, dir=None, name=None):
        """
        dummy constructor
        """
        if name is None:
            # construct a dummy class 
            self._dir = None
            self._name = None
        else: 
            # construct a named class 
            self._dir = dir
            self._name = name
        #
        self._width = None
        self._height = None
        self._fps = None
        self._time = None
        # 
        self._frames: List[asFrame] = []
        
        # video characterictics
        self._width = None
        self._height = None
        self._fps = None
        self._time = None
        self._cur_position = None  # Initialize these variables
        self.frame_counts = None
        self._duration = None  # Initialize duration attribute

    @property
    def dir(self):
        return self._dir

    @dir.setter
    def dir(self, value):
        self._dir = value

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, value):
        self._width = int( value )

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, value):
        self._height = int( value )

    @property
    def fps(self):
        return self._fps

    @fps.setter
    def fps(self, value):
        self._fps = int( value )

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._time = value

    @property
    def frames(self):
        return self._frames

    @frames.setter
    def frames(self, frames):
        self._frames = []
        for item in frames:
            self._frames.append(item)

    @property
    def cur_position(self):
        return self._cur_position

    @cur_position.setter
    def cur_position(self, value):
        self._cur_position = value

    @property
    def frame_counts(self):
        return self._frame_counts

    @frame_counts.setter
    def frame_counts(self, value):
        self._frame_counts = value

    @property
    def duration(self):
        if self._fps and self._frame_counts:
            return self._frame_counts / self._fps
        else:
            return None

    def init_video(self):
        """
        initialise video-related parameters: width, height, fps, time
        and 
        frames = []
        """
        import cv2 
        cap = cv2.VideoCapture( self.get_fullname() )
        #        
        self.width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH )
        self.height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT )
        self.fps    = cap.get(cv2.CAP_PROP_FPS)            # frames per second   
        #
        self.cur_position = cap.get(cv2.CAP_PROP_POS_MSEC)       # the current position of the video    
        self.frame_counts = cap.get(cv2.CAP_PROP_FRAME_COUNT)    # number of frames in the video file   

        cap.release()


    ##########################################
    ### Getters and Setters for attributes ###
    ##########################################
    def get_dir(self):
        return self._dir

    def set_dir(self, value):
        # self._dir = value
        pass

    def get_name(self):
        return self._name

    def set_name(self, value):
        # self._name = value
        pass

    def get_fullname(self):
        """
        get fullname: dir+name
        """
        return self._dir + "/" + self._name

    def get_name_without_extension(self):
        """
        Just the name of video without extension
        """
        import os        
        filename_without_extension, _ = os.path.splitext( self._name )
        return filename_without_extension

    def get_width(self):
        return self._width
        
    def set_width(self, value):
        self._width = int( value )
 
    def get_height(self):
        return self._height

    def set_height(self, value):
        self._height = int( value )        

    def get_fps(self):
        return self._fps

    def set_fps(self, value):
        self._fps = int( value )

    def get_time(self):
        return self._time

    def set_time(self, value):
        self._time = value
        

    def display_info(self, is_detailed=False):
        """
        Dispays the information about video
        """
        print()        
        print("----- Information about video")        
        print(f"Video Name: {self._name}")
        print(f"Resolution: {self._width} x {self._height}")
        print(f"FPS: {self._fps}")
        print(f"Time: {self._time} seconds")
        #
        if is_detailed:
            print("-----------the detailed infromation-------------")
            print(f"dir: {self._dir}")


    def process_annotated_frames(self, image_processor, dir_to_save):
        """
        processes all frames, create annotated and save them into path_to_save       
        """
        from tqdm import tqdm
        print("")
        print( "========================================================")
        print(f"=== Process annotated frames for movies {self._name} ===")
        print( "========================================================")        
        for frame in tqdm(self._frames):
            frame.save_annotated_frame(image_processor, dir_to_save)


    def create_video_annotated(self, dir_to_read_from, dir_write_to, to_show_frames=False):
        """
        create annotated video from frames in dir_to_read_from        
        """
        import os
        import cv2
        from tqdm import tqdm

        # create a path to read frames from
        path_read_from = "./" + dir_to_read_from + "/"
        print(path_read_from)

        # create a path to wrte an annotated movie video to
        path_write_to = "./" + dir_write_to + "/"

        # take all frames from that path
        file_list = os.listdir(path_read_from)

        # Sort the file list based on the extracted numbers
        sorted_files = sorted(file_list, key=self.__extract_number)

        #video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*'MPEG'), int( self._fps ), \
        #                                                ( int( self._width ), int( self._height ) ))

        # add '_videos_annotated' to the video filename
        filename, file_extension = os.path.splitext( self._name )
        # print( path_write_to + filename + "_" + dir_write_to + file_extension )
        video = cv2.VideoWriter( path_write_to + filename + "_" + dir_write_to + file_extension, \
                                0, 1, ( int( self._width ), int( self._height ) ) )
        #for annotation in annotations:
        print()
        print( "===================================================")
        print(f"=== Create the annotated movie for {self._name} ===")
        print( "===================================================")
        for filename in tqdm(sorted_files):
            
            #print(path_read_from + filename)
            annotated_frame = asFrame.read_frame(path_read_from + filename)
            # writing the new frame in output
            video.write(annotated_frame)

            # show frame if to_show_frames
            if to_show_frames:
                asFrame.show_frame( annotated_frame, text=f"Annotated frame {filename}")

        cv2.destroyAllWindows()        
        video.release()

    def create_video_segmentation_mask(self, dir_to_read_from, dir_write_to, to_show_frames=False):
        """
        create annotated video from frames in dir_to_read_from        
        """
        pass
        # FINISHED HERE 
        # TAKE A LOOK AT get_segmentation_mask in asFrame 
        # and at
        # create_video_annotated above  
        # to create the correpoding movie 


    def create_video_with_landmarks(self):
        """
        create video with landmarks from frames
        """
        pass

    def create_joined_video(self, other_video, is_reference=False):
        """
        create a joined video
        Input: 
        other_video : asVideo   
        is_reference=False    (in this case the second/parameter video is a reference)
        """
        pass 


    def print_frames(self):
        """
        prints names all the frames of the current video:
        """
        print()
        print(f"Printing all the frames of the video: {self.get_fullname()}")
        for item in self.frames:
            print(f"Frame: {item.name}")



    ###########################
    ### helpful subroutines ###
    ###########################
    def __extract_number(self, filename):
        try:
            # Extract numbers from the filename
            return int(''.join(filter(str.isdigit, filename)))
        except ValueError:
            return float('inf') 