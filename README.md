----
# PREDICTING CAR PRICES
----

IN THIS PROJECT WE WILL PREDICT CAR PRICES WITH A TECHNIQUE CALLED "K-NEAREST NEIGHBOR" OR KNN ALRGORITHM USING AUTOMOBILE DATASET FROM UCI MACHINE LEARNING REPOSOTORY

    - EXPLORING THE FUNDAMENTALS OF MACHINE LEARNING USING THE K_NEAREST NEIGHBORS ALOGORITHM

    - PRACTICING THE MACHINE LEARNING WORKFLOW TO PREDICT CAR'S MARTKET PRICE USING ITS ATTRIBUTES.

    - THE DATASET WE WILL BE WORKING WITH CONTAINS INFORMATION ON VARIOUS CARS.

![PICTURE](/Users/premsurawut/Github/DataSci_project/Machine_learning/knn/images/data_description.png)

----
## About KNN Regressor

KNN Regressor predicts the price of a given test observation by identifying the observations that are nearest to it. Because of 
this the scale of variables in such a dataset is very important. Variables on a large scale will have a larger effect on the 
-distance between the observations which also affects the KNN regressor too.

An intuitive way to handle the scaling problem in KNN regressor is to standardize the dataset in such a way that all variables given a mean of 0 and 1.

### Eucidean distance:

![PICTURE](/Users/premsurawut/Github/DataSci_project/Machine_learning/knn/images/eucidean_distance.png)

### Prediction algorithm:
----
    - Calculate the distance from x to all points in the dataset.

    - Sort the points in the dataset by ascending the distance.

    - Predict the majority label of the "K" Closest points.

### Evaluating model performance with MSE & RMSE:
----

![PICTURE](/Users/premsurawut/Github/DataSci_project/Machine_learning/knn/images/MSE.png)

![PICTURE](/Users/premsurawut/Github/DataSci_project/Machine_learning/knn/images/RMSE.png)

![PICTURE](/Users/premsurawut/Github/DataSci_project/Machine_learning/knn/images/RMSE1.png)

## Installation
----
```bash
!pip install heatmapz
```

## Important libraries
----
```
import pandas as pd
pd.options.display.max_columns = 99
import numpy as np

import seaborn as sns
sns.set(color_codes=True, font_scale=1.2)
from pylab import rcParams
rcParams['figure.figsize'] = 7,7 
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'
%load_ext autoreload
%autoreload 2
from heatmap import heatmap, corrplot

from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
```
