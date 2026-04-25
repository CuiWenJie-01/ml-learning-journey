import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.rc('font', size=14)
plt.rc('axes', labelsize=14, titlesize=14)
plt.rc('legend', fontsize=14)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)

housing_full = pd.read_csv('datasets/housing/housing.csv')

housing_full.hist(bins=50, figsize=(12, 8))
plt.show()