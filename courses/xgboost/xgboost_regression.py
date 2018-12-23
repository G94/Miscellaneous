#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Create the training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2, random_state=123)

# Instantiate the XGBRegressor: xg_reg
xg_reg = xgb.XGBRegressor(seed=123, objective="reg:linear", n_estimators=10)

# Fit the regressor to the training set
xg_reg.fit(X_train,y_train)

# Predict the labels of the test set: preds
preds = xg_reg.predict(X_test)

# Compute the rmse: rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))


#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

reg_params = [1, 10, 100]

# Create the initial parameter dictionary for varying l2 strength: params
params = {"objective":"reg:linear","max_depth":3}

# Create an empty list for storing rmses as a function of l2 complexity
rmses_l2 = []

# Iterate over reg_params
for reg in reg_params:

    # Update l2 strength
    params["lambda"] = reg
    
    # Pass this updated param dictionary into cv
    cv_results_rmse = xgb.cv(dtrain=housing_dmatrix, params=params, nfold=2, num_boost_round=5, metrics="rmse", as_pandas=True, seed=123)
    
    # Append best rmse (final round) to rmses_l2
    rmses_l2.append(cv_results_rmse["test-rmse-mean"].tail(1).values[0])

# Look at best rmse per l2 param
print("Best rmse as a function of l2:")
print(pd.DataFrame(list(zip(reg_params, rmses_l2)), columns=["l2", "rmse"]))

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------
# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data=X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear", "max_depth":2}

# Train the model: xg_reg
xg_reg = xgb.train(params=params, dtrain=housing_dmatrix, num_boost_round=10)

# Plot the first tree
xgb.plot_tree(xg_reg,num_trees = 0)
plt.show()

# Plot the fifth tree
xgb.plot_tree(xg_reg,num_trees = 4)
plt.show()

# Plot the last tree sideways
xgb.plot_tree(xg_reg,num_trees = 9,rankdir="LR")
plt.show()

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

# Create the DMatrix: housing_dmatrix
housing_dmatrix = xgb.DMatrix(data = X, label=y)

# Create the parameter dictionary: params
params = {"objective":"reg:linear","max_depth":4}

# Train the model: xg_reg
xg_reg = xgb.train(dtrain=housing_dmatrix, params = params, num_boost_round=10)

# Plot the feature importances
xgb.plot_importance(xg_reg)
plt.show()