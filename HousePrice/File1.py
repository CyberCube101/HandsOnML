''' Model output(a prediction of a districts median housing price
will be fed to another Machine Learning system, along with many other signals
This downstream system will determine whether it is worth investing in a given area'''
import numpy as np
import os
import tarfile
import urllib
import pandas as pd
import urllib.request
import matplotlib.pyplot as plt
from zlib import crc32
from sklearn.model_selection import train_test_split

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()

# print(housing.info())
# print(housing.head(2))
# print(housing['ocean_proximity'].value_counts())
# print(housing.describe())

print(housing.hist(bins=50, figsize=(20, 15)))
plt.show()


# we see a median income of 3 actually means $30,000
# we see the attributes have different scales
# Histograms are heavy tailed. This may make it harder for the ML algo

###### now create test/training set (20:80) using random rows in our data


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
print(len(housing))
print(len(test_set) / len(housing))


# to ensure that this works even with new dataset and we never use any of the test set
# in the dataset, we use hashing


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xfffffff < test_ratio * 2 ** 32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


# the housing dataset does not have an id column, so we just use row index

housing_with_id = housing.reset_index()

#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# use the train_test module in sklearn

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
