import configparser
import numpy as np
from pathlib import Path
config = configparser.ConfigParser()
config.read('config.ini')
a = Path(config['DEFAULT']['data'])
print(a.is_dir())