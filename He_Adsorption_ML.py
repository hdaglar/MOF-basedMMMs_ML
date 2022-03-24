import numpy as np
import pandas as pd
from sklearn.linear_model import LassoLarsCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from tpot.export_utils import set_param_recursive
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('D:/Fall2022_sept/TPOT_Thresholdlu exceller/Threshold/TOTAL _BUSON_excelleri/PHYTON_RAWDATA_FINAL/U_He_Submission.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('U_He (mol/kg)', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['U_He (mol/kg)'],train_size=0.80, test_size=0.20, random_state=42)

# Average CV score on the training set was: -2.8472954109115103e-06
exported_pipeline = make_pipeline(
    MinMaxScaler(),
    PolynomialFeatures(degree=2, include_bias=False, interaction_only=False),
    LassoLarsCV(normalize=False)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
y_pred_train=exported_pipeline.predict(training_features)
preds = exported_pipeline.predict(testing_features)


#ACCURACY
print('R2_Train: %.3f' % r2_score(training_target, y_pred_train))
print('R2_Test: %.3f' % r2_score(testing_target, preds))
print('MSE_Train: %.10f' % mean_squared_error(training_target, y_pred_train))
print('MSE_Test: %.10f' %mean_squared_error(testing_target, preds))
print('MAE_Train: %.10f' % mean_absolute_error(training_target, y_pred_train))
print('MAE_Test: %.10f' %mean_absolute_error(testing_target, preds))
mse_train = mean_squared_error(training_target, y_pred_train)
rmse_train = math.sqrt(mse_train)
mse_test = mean_squared_error(testing_target, preds)
rmse_test = math.sqrt(mse_test)

print('RMSE_Train: %.7f' % rmse_train)
print('RMSE_Test: %.7f' % rmse_test)

coef1, p = spearmanr(training_target, y_pred_train)
coef2, p = spearmanr(testing_target, preds)

print('SRCC_Train: %.3f' % coef1)
print('SRCC_Test: %.3f' % coef2)


plt.scatter(training_target, y_pred_train, color="blue")
plt.xlabel('truevalues_train')
plt.ylabel('predictedvalues_train')
plt.scatter(testing_target, preds, color="red")
plt.xlabel('Simulated')
plt.ylabel('ML-predicted')
plt.show()

# exctracted_best_model = exported_pipeline.steps[-1][1]
# exported_pipeline.fit(training_features, training_target) # X,y IMPORTNANT
# print(exctracted_best_model.feature_importances_)
# plt.xlabel('Feature Importance')
# feat_importances = pd.Series(exctracted_best_model.feature_importances_, index=testing_features.columns)
# feat_importances.nlargest(12).plot(kind='barh', color='orange')
# plt.show()

x_unseen = pd.read_csv('D:/Fall2022_sept/TPOT_Thresholdlu exceller/Threshold/TOTAL _BUSON_excelleri/PHYTON_RAWDATA_FINAL/500hMOFs.csv', sep=',', dtype=np.float64)
Y_unseen=exported_pipeline.predict(x_unseen)
df5 = pd.DataFrame(Y_unseen)
df5.to_csv('500hMOFs_HE_UPT.csv', index=False)