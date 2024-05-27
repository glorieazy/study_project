# from pose_extraction import *
# from syncronizer import *
# from process import *
# from time_sync import *


if __name__ == '__main__':


    ###################################
    ### Decompose video into frames ###
    ###################################
    from test_files.asVideo_file import asVideo
    FILES = [
        #Video("1", False,"media/mtest3.mp4"),
        #asVideo("1", False,"media/efe_serve.mp4"),
        #asVideo("2", True,"media/alcaraz_serve-synced.mp4")
        #
        #Video("1", False,"media/efe_serve.mp4"),
        #Video("2", True,"media/alcaraz_serve-synced.mp4")
        asVideo("1", False,"media/Daria_forhand.mp4"),
        asVideo("2", True,"media/Teana_forhand.mp4")
    ]

    from time_sync import *
    weight = serve    
    #shape = (1920,1080)
    shape = (1600, 720)

    FILES[0].as_deconstruct(shape, './test_data/')    
    # # #FILES[1].as_deconstruct(shape, './test_data/')    


    # # ###########################
    # # ### initialise detector ###
    # # ###########################
    from pose_extraction import *
    detector = configure()

    # # #########################
    # # ### Process the image ###
    # # #########################
    # FILES[0].as_process(detector)

    # ###########################
    # ### my test with images ###
    # ###########################
    # FILES[0].as_show_image('test_data/frame25.jpg')

    ########################
    ### chgeck landmarks ###
    ########################
    detection_result = FILES[0].as_process_image('test_data/frame36.jpg', detector)

    FILES[0].as_show_segmentation_mask(detection_result)
    print()
