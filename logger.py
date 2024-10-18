import logging
import os

import logging.handlers

class Logger:
    def __init__(self, log_dir='/var/log/voxnet', log_file='app.log', log_level=logging.DEBUG, backup_count=5):
        # Ensure the log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_path = os.path.join(log_dir, log_file)
        
        self.logger = logging.getLogger('VoxNetLogger')
        self.logger.setLevel(log_level)

        # Create a rotating file handler
        handler = logging.handlers.RotatingFileHandler(
            log_path, backupCount=backup_count)
        handler.setLevel(log_level)

        # Create a console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Create a formatter and set it for both handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # Add the handlers to the logger
        self.logger.addHandler(handler)
        self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger

# Usage example
if __name__ == "__main__":
    logger = Logger().get_logger()
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")