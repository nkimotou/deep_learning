from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

# Create and train a Lasso regression model
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)

# Make predictions
predictions_lasso = lasso_model.predict(X_test)

# Calculate mean squared error
mse_lasso = mean_squared_error(y_test, predictions_lasso)
print("MSE: ", mse_lasso)
