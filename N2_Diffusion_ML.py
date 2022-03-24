import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/D_N2_Submission.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('D_N2 (cm2/s)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['D_N2 (cm2/s)'],train_size=0.80, test_size=0.20, random_state=42)

exported_pipeline = GradientBoostingRegressor(alpha=0.95, learning_rate=0.1, loss="ls", max_depth=10, max_features=0.1, min_samples_leaf=6, min_samples_split=12, n_estimators=100, subsample=0.9000000000000001)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train=exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)


#PLOTTING
plt.scatter(training_target, y_pred_train, color="blue")
plt.xlabel('truevalues_train')
plt.ylabel('predictedvalues_train')
plt.scatter(testing_target, preds, color="red")
plt.xlabel('Simulated')
plt.ylabel('ML-predicted')
plt.show()


