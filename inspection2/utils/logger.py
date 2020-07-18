# -*- coding: utf-8 -*-

# Header ...

import os, logging


class Logger(object):
    def __init__(self, logger=None, name="log", log_dir=None):
        self.logger = None
        self.name = name
        self.log_dir = log_dir
        self.config_logger(logger)
        
    def config_logger(self, logger):
        if logger is not None: self.logger = logger
        elif self.log_dir is not None:       # Config the default logger
            logger = logging.getLogger(__name__)
            logger.setLevel(level = logging.INFO)
            handler = logging.FileHandler(self.name + ".log")
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            self.logger = logger

    def info(self, message, is_write=True):
        if is_write and self.logger is not None: self.logger.info(message)
        print("INFO - " + message)
        
    def debug(self, message, is_write=True):
        if is_write and self.logger is not None: self.logger.debug(message)
        print("DEBUG - " + message)
        
    def warning(self, message, is_write=True):
        if is_write and self.logger is not None: self.logger.warning(message)
        print("WARNING - " + message)
        
    def critical(self, message, is_write=True):
        if is_write and self.logger is not None: self.logger.critical(message)
        print("CRITICAL - " + message)
        
    def error(self, message, is_write=True):
        if is_write and self.logger is not None: self.logger.error(message)
        raise Exception(message)

if __name__ == "__main__":
    logger = Logger(name="log", log_dir=os.getcwd())
    logger.info("Test ...")
        