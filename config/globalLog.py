#globalLog.py
import logging
import logging.config
import os
  
def get_logger(name='root'):
    conf_log = os.path.abspath(os.getcwd() + "/config/logger_config.ini")
    logging.config.fileConfig(conf_log)
    return logging.getLogger(name)
 
 
logger = get_logger(__name__)
