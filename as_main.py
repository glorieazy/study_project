#################################
### The main executable file ####
#################################

def main():
    ######################
    ### starts manager ###
    ######################
    from as_Videos_Manager import asVideosManager
    manager = asVideosManager()

    #####################
    ### create config ###
    #####################
    from as_config import asConfig
    config_json = asConfig("config/config.json")
    # add it to manager
    manager.add_config( config_json )

    ##################
    ### Read video ###
    ##################    
    manager.set_video(type_of_video="primary")
    
    ############################
    ### Set video parameters ###
    ############################    
    manager.init_param_video(type_of_video="primary")

    ###################################
    ### Decompose video into frames ###
    ###################################
    manager.decompose_video_into_frames(type_of_video="primary")

    ################################
    ### Dump/restore the manager ###
    ################################
    # dump manager
    asVideosManager.dump( manager )
    # load manager
    manager = asVideosManager.load()
    print()

    manager.primary.frames[0].display_info()
    manager.primary.frames[-1].display_info()


    ################################
    ### configure ImageProcessor ###
    ################################
    from as_ImageProcessor import asImageProcessor
    image_processor = asImageProcessor()
    image_processor.initialize_detector()
    # add it to manager
    manager.add_image_processor( image_processor )

    ######################
    ### process Frames ###
    ######################
    #image_processor.process_frame( "./frames/Daria_forhand.mp4_frame12.jpg" )
    #print( "./frames/Daria_forhand.mp4_frame0.jpg" )
    
    #manager.primary.frames[0].show_annotated_frame(image_processor)        
    manager.primary.create_annotated_from_frames(image_processor, manager.config.get_value("dirs","frames_annotated"))

    # created annotated movie from annotated frames in a spesific directory
    manager.primary.create_video_annotated( manager.config.get_value("dirs","frames_annotated"), \
                                            manager.config.get_value("dirs","videos_annotated"), to_show_frames=False)    

    #manager.primary._frames[0].show()

    #from as_Frame import asFrame
    #asFrame.show_frame( manager.primary._frames[0].get_segmentation_mask(),  text= manager.primary._frames[0].name)
    #manager.primary._frames[0].show_segmentation_mask()

    print()
    # test 
    # resize frame


# Check if this script is being run directly (not imported as a module)
if __name__ == "__main__":
    main()  # Call the main function



    ####################
    ### resize frame ###
    ####################
    # manager.primary.frames[0].show()
    # shape = (1000, 800)
    # manager.primary.frames[0].resize(shape, to_change_file=True)
    # manager.primary.frames[0].show()
    # print()

