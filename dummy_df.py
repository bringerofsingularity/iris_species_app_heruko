import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import make_classification

features_array, target_array = make_classification(n_samples=5000, n_features=2, n_informative=2,
                           n_redundant=0, n_repeated=0, n_classes=3,
                           n_clusters_per_class=1,
                           weights=[0.02, 0.06, 0.92],
                           class_sep=0.8, random_state=0)

dummy_dict = {"col 1": [features_array[i][0] for i in range(features_array.shape[0])],
              "col 2": [features_array[i][1] for i in range(features_array.shape[0])],
              "target": target_array}

dummy_df = pd.DataFrame.from_dict(dummy_dict)

plt.figure(figsize = (8, 8))
sns.scatterplot(dummy_df["col 1"], dummy_df["col 2"], hue = dummy_df["target"])
plt.show()