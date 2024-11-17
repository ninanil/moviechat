import logging

def get_logger(name):
    """
    Create and configure a logger with a shared format.
    :param name: Name of the logger (usually the module/class name).
    :return: Configured logger.
    """
    logger = logging.getLogger(name)

    # Avoid adding multiple handlers if the logger already exists
    if not logger.hasHandlers():
        # Create a console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # Add the handler to the logger
        logger.addHandler(console_handler)

        # Set logging level
        logger.setLevel(logging.INFO)

    return logger
