import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, train_test_split

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')
print(train_data.info())

# date time conversion
train_data.datetime = pd.to_datetime(train_data.datetime)
train_data['hour'] = train_data.datetime.dt.hour
train_data['month'] = train_data.datetime.dt.month
train_data['year'] = train_data.datetime.dt.year

test_data.datetime = pd.to_datetime(test_data.datetime)
test_data['hour'] = test_data.datetime.dt.hour
test_data['month'] = test_data.datetime.dt.month
test_data['year'] = test_data.datetime.dt.year

# Dividing the features into categorical and numerical
categorical_features = ['season', 'holiday', 'year', 'month', 'hour', 'workingday', 'weather']
number_features = ['temp', 'atemp', 'humidity', 'windspeed']
# Our target here is to estimate the total count of users on any given particular hour
target_features = ['registered', 'casual']

for col in categorical_features:
    train_data[col] = train_data[col].astype('category')
print(train_data[categorical_features].describe())
print(train_data[number_features].describe())

# Check if any null values are present in data
print(train_data.isnull().any())

sn.set(font_scale=1.0)
fig, ax = plt.subplots(1, 2)
fig.set_size_inches(10, 5)
sn.boxplot(data=train_data, y = 'registered', ax=ax[0])
ax[0].set(ylabel= 'Count', title = 'Registered')
sn.boxplot(data=train_data, y = 'casual', ax=ax[1])
ax[1].set(ylabel= 'Count', title = 'Casual')
plt.show()



fig, ax = plt.subplots(1, 2)
fig.set_size_inches(20, 5)
sn.boxplot(data=train_data, y = 'count', x = 'year', ax=ax[0])
ax[0].set(xlabel = 'Year', ylabel= 'Count', title = 'Count Over Years')
sn.boxplot(data=train_data, y = 'count', x = 'month', ax=ax[1])
ax[1].set(xlabel='Month', ylabel= 'Count', title = 'Count Over Months')
plt.show()


fig, ax = plt.subplots(1, 2)
fig.set_size_inches(20, 5)
sn.boxplot(data=train_data, y = 'count', x = 'weather', ax=ax[0])
ax[0].set(xlabel='Weather', ylabel= 'Count', title = 'Count Over Weather Condition')
sn.boxplot(data=train_data, y = 'count', x = 'workingday', ax=ax[1])
ax[1].set(xlabel='Working Day', ylabel= 'Count', title = 'Count Over Working/ Non-Working Day')
plt.show()


fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sn.boxplot(data=train_data, y = 'count', x = 'temp', ax=ax)
ax.set(xlabel='Temperature', ylabel= 'Count', title = 'Count Over Temperatures')
plt.show()


fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sn.boxplot(data=train_data, y = 'count', x = 'humidity', ax=ax)
ax.set(xlabel='Humidity', ylabel= 'Count', title = 'Count Over Humidity')
plt.show()


fig, ax = plt.subplots()
fig.set_size_inches(20, 5)
sn.boxplot(data=train_data, y = 'count', x = 'windspeed', ax=ax)
ax.set(xlabel='Wind Speed', ylabel= 'Count', title = 'Count Over Wind Speeds')
plt.show()


fig, ax = plt.subplots(3, 2)
fig.set_size_inches(30, 30)
sn.boxplot(data=train_data[train_data.workingday == 1], y = 'count', x = 'hour', ax=ax[0][0])
ax[0][0].set(xlabel='Hour', ylabel= 'Count', title = 'Count Over Hour of the Working Day')
sn.boxplot(data=train_data[train_data.workingday == 0], y = 'count', x = 'hour', ax=ax[0][1])
ax[0][1].set(xlabel='Hour', ylabel= 'Count', title = 'Count Over Hour of the Weekend')

sn.boxplot(data=train_data[train_data.workingday == 1], y = 'registered', x = 'hour', ax=ax[1][0])
ax[1][0].set(xlabel='Hour', ylabel= 'Reg Count', title = 'Registered Users Over Hour of the Working Day')
sn.boxplot(data=train_data[train_data.workingday == 0], y = 'registered', x = 'hour', ax=ax[1][1])
ax[1][1].set(xlabel='Hour', ylabel= 'Reg Count', title = 'Registered Users Over Hour of the Weekend')

sn.boxplot(data=train_data[train_data.workingday == 1], y = 'casual', x = 'hour', ax=ax[2][0])
ax[2][0].set(xlabel='Hour', ylabel= 'Casual Count', title = 'Casual Users Over Hour of the Working Day')
sn.boxplot(data=train_data[train_data.workingday == 0], y = 'casual', x = 'hour', ax=ax[2][1])
ax[2][1].set(xlabel='Hour', ylabel= 'Casual Count', title = 'Casual Users Over Hour of the Weekend')
plt.show()


fig, ax =plt.subplots(1,2)
fig.set_size_inches(10, 5)
sn.distplot(train_data.registered, ax=ax[0], label='Registered')
sn.distplot(train_data.casual, ax=ax[1], label='Casual')
plt.show()


fig, ax = plt.subplots(figsize=(15,10))
sn.heatmap(train_data[number_features + target_features].corr(), linewidths=1, square=True, annot=True, cmap="RdYlGn", ax=ax)
plt.show()


number_features.remove('atemp')

def RMLSE(y_true, y_pred):
    return np.sqrt(np.mean(np.square(np.log(y_true + 1) - np.log(y_pred + 1))))
def Accuracy(y_true, y_pred):
    return 100*(1 - np.mean(abs(y_pred - y_true)/y_true))
def RunModel(model, params, X_train, X_test, y_train):
    fit_mod = GridSearchCV(model, params, cv=5, scoring='neg_mean_squared_error')
    fit_mod.fit(X_train, y_train)
    y_predict = fit_mod.predict(X_test)
    return y_predict, fit_mod
def plot_cv(params, bestreg, variable):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(params[variable],bestreg.cv_results_['mean_test_score'],'o-')
    plt.xlabel(variable)
    plt.ylabel('score mean')
    plt.subplot(122)
    plt.plot(params[variable],bestreg.cv_results_['std_test_score'],'o-')
    plt.xlabel(variable)
    plt.ylabel('score std')
    plt.tight_layout()
    plt.show()


X = train_data[categorical_features+number_features]
y_tot = train_data['count']
y_cas = train_data['casual']
y_reg = train_data['registered']

X_train_cas, X_test_cas, y_train_cas, y_test_cas = train_test_split(X, y_cas, test_size=0.1, random_state=0)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.1, random_state=0)

#Fitting both casual and registered users with Random Forest Regression model
reg_rf = RandomForestRegressor()
params_rf = {
    "n_estimators": [200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500]
    }

y_cas_rf, mod_cas_rf = RunModel(reg_rf, params_rf, X_train_cas, X_test_cas, y_train_cas)
y_reg_rf, mod_reg_rf = RunModel(reg_rf, params_rf, X_train_reg, X_test_reg, y_train_reg)

print(mod_cas_rf.best_params_)
print(mod_reg_rf.best_params_)

plot_cv(params_rf, mod_cas_rf, 'n_estimators')
plot_cv(params_rf, mod_reg_rf, 'n_estimators')

reg_rf = RandomForestRegressor()
params_rf = {
    "min_samples_leaf": np.arange(1,10,1)}

y_cas_rf, mod_cas_rf = RunModel(reg_rf, params_rf, X_train_cas, X_test_cas, y_train_cas)
y_reg_rf, mod_reg_rf = RunModel(reg_rf, params_rf, X_train_reg, X_test_reg, y_train_reg)

print(mod_cas_rf.best_params_)
print(mod_reg_rf.best_params_)

plot_cv(params_rf, mod_cas_rf, "min_samples_leaf")
plot_cv(params_rf, mod_reg_rf, "min_samples_leaf")

reg_rf = RandomForestRegressor()
params_cas_rf = {
    "n_estimators": [900],
    "min_samples_leaf": [1]}
params_reg_rf = {
    "n_estimators": [600],
    "min_samples_leaf": [2]}

y_cas_rf, mod_cas_rf = RunModel(reg_rf, params_cas_rf, X_train_cas, X_test_cas, y_train_cas)
y_reg_rf, mod_reg_rf = RunModel(reg_rf, params_reg_rf, X_train_reg, X_test_reg, y_train_reg)

y_cas_rf[y_cas_rf < 0] = 0
y_reg_rf[y_reg_rf < 0] = 0
y_pred_cas_rf = mod_cas_rf.predict(X)
y_pred_reg_rf = mod_reg_rf.predict(X)
y_pred_cas_rf[y_pred_cas_rf < 0] = 0
y_pred_reg_rf[y_pred_reg_rf < 0] = 0
y_tot_rf = np.round(y_pred_cas_rf + y_pred_reg_rf)

#Random Forest Regressor
print('Root Mean Log Squared Error of the Casual test model is ', RMLSE(y_test_cas, y_cas_rf))
print('Root Mean Log Squared Error of the Registered test model is ', RMLSE(y_test_reg, y_reg_rf))

print('Root Mean Log Squared Error of the model is ', RMLSE(y_tot, y_tot_rf))
print('Accuracy of the model is ', Accuracy(y_tot, y_tot_rf))

#Fitting both casual and registered users with Gradient Boost Regression model
reg_gb = GradientBoostingRegressor()
params_gb = {
    "n_estimators": [600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
    }
y_cas_gb, mod_cas_gb = RunModel(reg_gb, params_gb, X_train_cas, X_test_cas, y_train_cas)
y_reg_gb, mod_reg_gb = RunModel(reg_gb, params_gb, X_train_reg, X_test_reg, y_train_reg)

print(mod_cas_gb.best_params_)
print(mod_reg_gb.best_params_)

plot_cv(params_gb, mod_cas_gb, 'n_estimators')
plot_cv(params_gb, mod_reg_gb, 'n_estimators')

reg_gb = GradientBoostingRegressor()
params_gb = {
    "min_samples_leaf": np.arange(1,10,1)
    }
y_cas_gb, mod_cas_gb = RunModel(reg_gb, params_gb, X_train_cas, X_test_cas, y_train_cas)
y_reg_gb, mod_reg_gb = RunModel(reg_gb, params_gb, X_train_reg, X_test_reg, y_train_reg)

print(mod_cas_gb.best_params_)
print(mod_reg_gb.best_params_)

plot_cv(params_gb, mod_cas_gb, "min_samples_leaf")
plot_cv(params_gb, mod_reg_gb, "min_samples_leaf")

reg_gb = GradientBoostingRegressor()
params_cas_gb = {
    "n_estimators": [2000],
    "min_samples_leaf": [8]
    }
params_reg_gb = {
    "n_estimators": [2000],
    "min_samples_leaf": [6]
    }
y_cas_gb, mod_cas_gb = RunModel(reg_gb, params_cas_gb, X_train_cas, X_test_cas, y_train_cas)
y_reg_gb, mod_reg_gb = RunModel(reg_gb, params_reg_gb, X_train_reg, X_test_reg, y_train_reg)

y_cas_gb[y_cas_gb < 0] = 0
y_reg_gb[y_reg_gb < 0] = 0
y_pred_cas_gb = mod_cas_gb.predict(X)
y_pred_reg_gb = mod_reg_gb.predict(X)
y_pred_cas_gb[y_pred_cas_gb < 0] = 0
y_pred_reg_gb[y_pred_reg_gb < 0] = 0
y_tot_gb = np.round(y_pred_cas_gb + y_pred_reg_gb)

print('Root Mean Log Squared Error of the Casual test model is ', RMLSE(y_test_cas, y_cas_gb))
print('Root Mean Log Squared Error of the Registered test model is ', RMLSE(y_test_reg, y_reg_gb))

print('Root Mean Log Squared Error of the model is ', RMLSE(y_tot, y_tot_gb))
print('Accuracy of the model is ', Accuracy(y_tot, y_tot_gb))

#Random Forest Regressor
print('Root Mean Log Squared Error of the Random Forest Regressor model is ', RMLSE(y_tot, y_tot_rf))
print('Accuracy of the Random Forest Regressor model is ', Accuracy(y_tot, y_tot_rf))


#Gradient Boost Regressor
print('Root Mean Log Squared Error of the Gradient Boost Regressor model is ', RMLSE(y_tot, y_tot_gb))
print('Accuracy of the Gradient Boost Regressor model is ', Accuracy(y_tot, y_tot_gb))


fig, ax = plt.subplots(figsize=(15, 5))
plt.bar(categorical_features+number_features, mod_reg_rf.best_estimator_.feature_importances_)
plt.title('Registered Users Feature Importance')
plt.show()


fig, ax = plt.subplots(figsize=(15, 5))
plt.bar(categorical_features+number_features, mod_cas_rf.best_estimator_.feature_importances_)
plt.title('Casual Users Feature Importance')
plt.show()


plt.hist(y_tot, bins=30, alpha=0.5, color='g', label='Actual')
plt.hist(y_tot_rf, bins=30, alpha=0.5, color='b', label='Final predict')
plt.legend(loc='upper right')
plt.show()


X_test = test_data[categorical_features+number_features]
y_test_cas_rf = mod_cas_rf.predict(X_test)
y_test_cas_rf[y_test_cas_rf < 0] = 0
y_test_reg_rf = mod_reg_rf.predict(X_test)
y_test_reg_rf[y_test_reg_rf < 0] = 0
y_test = y_test_cas_rf + y_test_reg_rf


test_data['count'] = y_test
sn.distplot(test_data['count'])
df_out = pd.DataFrame(test_data[['datetime', 'count']])
df_out.head()
df_out.to_csv('output.csv', index= False)