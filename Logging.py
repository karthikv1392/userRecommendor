_Author_ = "Karthik Vaidhyanathan"

import logging.handlers
import time
from ConfigParser import SafeConfigParser

CONFIGURATION_FILE = "settings.conf"
parser = SafeConfigParser()
parser.read(CONFIGURATION_FILE)
BASEPATH = parser.get('settings','log_path')
log_level = parser.get('settings','log_level')
LOG_FILE = BASEPATH  + "recommendor"

logger = logging.getLogger('recommendor')
logger_file = logging.FileHandler(LOG_FILE + "_" + time.strftime("%Y%m%d")+'.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s',"%Y/%m/%d %H:%M:%S")
logger_file.setFormatter(formatter)
logger.addHandler(logger_file)
logger.setLevel(log_level)
logger.debug('Logger Initialized')
