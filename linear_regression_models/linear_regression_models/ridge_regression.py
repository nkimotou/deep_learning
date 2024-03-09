from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Create and train a ridge regression model
ridge_model = Ridge(alpha=0.1)
ridge_model.fit(X_train, y_train)

# Make predictions
predictions_ridge = ridge_model.predict(X_test)

# Calculate mean squared error
mse_ridge = mean_squared_error(y_test, predictions_ridge)
print("MSE: ", mse_ridge)
