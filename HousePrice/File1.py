import os
import tarfile
import urllib
import pandas as pd

this_dir = os.getcwd()

DOWNLOAD_ROOT = "https://github.com/ageron/handson-ml2/tree/master/"
HOUSING_PATH = this_dir
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"
