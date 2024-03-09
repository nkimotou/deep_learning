from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Create and train an ElasticNet regression model
elastic_net_model = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic_net_model.fit(X_train, y_train)

# Make predictions
predictions_elastic_net = elastic_net_model.predict(X_test)

# Calculate mean squared error
mse_elastic_net = mean_squared_error(y_test, predictions_elastic_net)
print("MSE: ", mse_elastic_net)
