from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# Transform features to include polynomial terms up to degree 2
poly = PolynomialFeatures(degree=2)
X = poly.fit_transform(X)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a linear regression model with polynomial features
model_poly = LinearRegression()
model_poly.fit(X_train, y_train)

# Make predictions
predictions_poly = model_poly.predict(X_test)

# Calculate mean squared error
mse_poly = mean_squared_error(y_test, predictions_poly)
print("MSE: ", mse_poly)
