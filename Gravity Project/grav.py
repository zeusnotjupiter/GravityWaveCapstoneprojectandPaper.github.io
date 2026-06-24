#SVM took way too long to train, so I didn't attempt to include it. I settled on linear regression, lasso regression, and random forest regression for the final model.
import pandas as pd
import tarfile
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
# here I am loading the datasets that were extracted. Labeling them so I can combine them later.
# sep=r'\s+' tells pandas that columns are separated by whitespace characters..
df1 = pd.read_csv('LEE.stuff/waveform1BHNS.out', sep=r'\s+', header=None, skiprows=1, engine='python')
df2 = pd.read_csv('LEE.stuff/waveform2BHNS.out', sep=r'\s+', header=None, skiprows=1, engine='python')
df3 = pd.read_csv('LEE.stuff/waveform3BHNS.out', sep=r'\s+', header=None, skiprows=1, engine='python')
df4 = pd.read_csv('LEE.stuff/waveform4BHNS.out', sep=r'\s+', header=None, skiprows=1, engine='python')
df5 = pd.read_csv('LEE.stuff/waveform5BHNS.out', sep=r'\s+', header=None, skiprows=1, engine='python')
#I am adding features based on the conditions of each simulation

# Had to give column names for all datasets because they were not included in the original files.
for df in [df1, df2, df3, df4, df5]:
    df.columns = ['t (sec)', 'rh+ (cm)', 'rhx (cm)']
# r is the distance from the observer^.
# print (df) #<----uncomment this line to see the raw data
# I Combined all of the datasets vertically into a single DataFrame.
dfcombined = pd.concat([df1, df2, df3, df4], ignore_index=True)
dfcombined2 = pd.concat([df1, df2, df3], ignore_index=True)
dfcombined.to_csv('combined_gravitational_wave_data.csv', index = False) #run this line only once to create the file
#
#
dfc = pd.read_csv('combined_gravitational_wave_data.csv')
X_train = (dfcombined[['t (sec)', 'rh+ (cm)']].values)
y_train = (dfcombined['rhx (cm)'].values)
X_test = (df5[['t (sec)', 'rh+ (cm)']].values)
y_test = ((df5['rhx (cm)']).values) 
#since I am training the model to simply predict rhx purely based on time and rh+ amplitude, I don't need to add any additional features.
from sklearn.linear_model import LinearRegression, Lasso
#from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
lin_reg = LinearRegression()
lasso_reg = Lasso()
#got this from DTSC 670
def rsmedoublenegative(model, X, y):
    return np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv = 3))
linreg = rsmedoublenegative(lin_reg, X_train, y_train)
lassoreg = rsmedoublenegative(lasso_reg, X_train, y_train)
#RSME
linregmean = linreg.mean()
lassoregmean = lassoreg.mean()
#standard deviation
linregstd = linreg.std()
lassoregstd = lassoreg.std()
#r^2 scores
lin_r2 = cross_val_score(lin_reg, X_train, y_train, scoring='r2', cv=3).mean()
lasso_r2 = cross_val_score(lasso_reg, X_train, y_train, scoring='r2', cv=3).mean()
print("Linear Regression RMSE:", linregmean) 
print("Lasso Regression RMSE:", lassoregmean)
print("Linear Regression RMSE Std Dev:", linregstd)
print("Lasso Regression RMSE Std Dev:", lassoregstd)
print("Linear Regression R^2:", lin_r2)
print("Lasso Regression R^2:", lasso_r2)
print("rhx min:", dfc['rhx (cm)'].min())
print("rhx max:", dfc['rhx (cm)'].max())



#RandomForestRegressor has always been my go to for non-linear learning. I wouldve used a CNN but thats too complex for this project in my opinion considering low amount of features.
from sklearn.ensemble import RandomForestRegressor
#100 trees always works for me, so thats why I chose that number specifically, I won't claim to know the science behind it.
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
rf_pred = rf_reg.predict(X_test)
rf_r2 = r2_score(y_test, rf_pred)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
print("Random Forest RMSE:", rf_rmse)
print("Random Forest R^2:", rf_r2)
print(rf_pred)
#RMSE=[p;l,m]
#rf_rmse = np.sqrt(mean_squared_error(y_test, rf_predictions))
#r^2
#rf_r2 = r2_score(y_test, rf_predictions)
#print("Random Forest RMSE:", rf_rmse)
#print("Random Forest R^2:", rf_r2)

#testcombined = pd.concat([df4,df5], ignore_index=True)

#X_train2 = (dfcombined2[['t (sec)', 'rh+ (cm)']].values)
#y_train2 = (dfcombined2['rhx (cm)'].values)
#X_test2 = (testcombined[['t (sec)', 'rh+ (cm)']].values)
#y_test2 = ((testcombined['rhx (cm)']).values) 

#randomforest 2nd model with different parameters, I wanted to see if it would perform better than the default parameters.
#rf_reg2 = RandomForestRegressor(n_estimators=2000, random_state=42)
#rf_reg2.fit(X_train2, y_train2)
#rf_pred2 = rf_reg2.predict(X_test2)
#rf_r2_2 = r2_score(y_test2, rf_pred2)
#rf_rmse_2 = np.sqrt(mean_squared_error(y_test2, rf_pred2))
#print("Random Forest 2 RMSE:", rf_rmse_2)
#print("Random Forest 2 R^2:", rf_r2_2)

#xgboost 2nd model with different parameters, I wanted to see if it would perform better than the default parameters.
#model2 = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=10, random_state=42)
#model2.fit(X_train, y_train)
#xgb_pred2 = model2.predict(X_test)
#xgb_rmse_2 = np.sqrt(mean_squared_error(y_test, xgb_pred2))
#xgb_r2_2 = r2_score(y_test, xgb_pred2)
#print("XGBoost 2 RMSE:", xgb_rmse_2)
#print("XGBoost 2 R^2:", xgb_r2_2)

#model file
joblib.dump(rf_reg, 'gravitational_wave_model.pkl')
pd.DataFrame(rf_pred)
df = pd.DataFrame(rf_pred)
df.to_csv('prediction.csv', sep=';', index=False)

#Contextualizing the results because RMSE doesn't mean much without the range.
#print(f"rhx values range from {y.min():.2f} to {y.max():.2f}")
#print(f"RMSE of {rf_rmse:.2f} represents {(rf_rmse/y.std())*100:.1f}% of standard deviation")
#print(f"R^2 of {rf_r2:.3f} means a {rf_r2*100:.1f}% fit of the data")
# Scikit-learn Documentation: https://scikit-learn.org/stable/, Kaggle Learn: Introduction to Machine Learning: https://www.kaggle.com/learn/intro-to-machine-learning, Python Data Science Libraries Documentation (NumPy, Pandas, Matplotlib, Joblib) for functions
#Lee W.H., 2001, Newtonian hydrodynamics of the coalescence of black holes with neutron stars -- IV. Irrotational binaries with a soft equation of state, Monthly Notices of the Royal Astronomical Society, Volume 328, Issue 2, Pages 583-600, https://doi.org/10.1046/j.1365-8711.2001.04898.x

