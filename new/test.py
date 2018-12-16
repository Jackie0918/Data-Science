# Import necessary modules for data analysis and data visualization.
# Data analysis modules
import pandas as pd

# numpy is a great library for doing mathmetical operations.
import numpy as np

# Visualization libraries
from matplotlib import pyplot as plt
import seaborn as sns
from IPython import get_ipython

# get_ipython().run_line_magic('matplotlib', 'inline')

## Machine learning libraries
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.model_selection import cross_val_score,GridSearchCV

## Ignore warning
import warnings
warnings.filterwarnings('ignore')

## Importing the datasets
train = pd.read_csv("./input/train.csv")
test = pd.read_csv("./input/test.csv")
print(train)