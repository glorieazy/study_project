"""
this is the logging module
To use:
    logger = CreateLogger(filename=None)
    logger.debug('This is a debug Xmessage')
    logger.info('This is an info Xmessage value=%s', value)
    logger.warning('This is a warning Xmessage')
    logger.error('This is an error message')
    logger.critical('This is a critical message')
    try:
        # Your code that may raise an exception
        result = 1 / 0
    except ZeroDivisionError as e:
        # Log the exception
        logger.exception("An error occurred: %s", e)
    
    logger.exception("An error occurred: %s", Exception("It is my Exception"))    
"""

def CreateLogger(filename=None):
    """
    Creates the logger
    Output:
        logger
    """
    import logging
    from datetime import datetime

    # create a filename
    if filename is None:
        today = datetime.now().today()
        filename = f"log/log{today.day}-{today.month}-{today.year}.txt"
    # clean the loggin file        
    open(filename, "w").close()
    
    # logging parameters
    format_str = '%(asctime)-9s [%(levelname)-8s] <-%(funcName)-5s-> %(message)s'
    datefmt='%d-%m-%Y %H:%M:%S'
    mode='w'

    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    

    # Create file handler
    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setLevel(logging.DEBUG)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)  

    # Create a formatter and set it for both handlers
    formatter = logging.Formatter( format_str, datefmt=datefmt )                  
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add both handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger



####################
### Test example ###
####################
def main():

    logger = CreateLogger()

    value = 42
    logger.debug('This is a debug Xmessage')
    logger.info('This is an info Xmessage value=%s', value)
    logger.warning('This is a warning Xmessage')
    logger.error('This is an error message')
    logger.critical('This is a critical message')

    try:
        # Your code that may raise an exception
        result = 1 / 0
    except ZeroDivisionError as e:
        # Log the exception
        logger.exception("An error occurred: %s", e)
    
    logger.exception("An error occurred: %s", Exception("It is my Exception"))


if __name__ == '__main__':
    main()
    