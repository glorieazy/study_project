from pose_extraction import *
from syncronizer import *
from process import *
from time_sync import *


if __name__ == '__main__':
    FILES = [
        #Video("1", False,"media/mtest3.mp4"),
        #Video("1", False,"media/efe_serve.mp4"),
        #Video("2", True,"media/alcaraz_serve-synced.mp4")
        #        
        Video("1", False,"media/Teana_forhand.mp4"),
        Video("2", True,"media/Daria_forhand.mp4")       
    ]

    #weight = serve    
    #shape = (1920,1080)

    weight = serve
    shape = (1600,720)

    FILES[0].deconstruct(shape)    
    FILES[1].deconstruct(shape)

    detector = configure()

    FILES[0].process(detector)
    FILES[1].process(detector)

    ref = get_shorter(FILES)
    FILES[0].reference_length_candidates, FILES[0].position_candidates = pre_normalize(FILES[0], ref)
    FILES[1].reference_length_candidates, FILES[1].position_candidates = pre_normalize(FILES[1], ref)

    #TIME SYNCRONIZER
    frames1, frames2 = pre_sync(FILES, ref)
    modify_results(FILES, ref, frames1, frames2, weight)

    set_average_length(FILES)

    video_annotate(detector, FILES[0], find_min(FILES), solutions.drawing_utils.WHITE_COLOR, shape)
    video_make(FILES[0])
    video_convert(FILES[0], ref)
    clean()

    video_annotate(detector, FILES[1], find_min(FILES), solutions.drawing_utils.BLUE_COLOR, shape)
    video_make(FILES[1])
    video_convert(FILES[1], ref)
    clean()

    blend(FILES, find_min(FILES))

    print("So far, so good")