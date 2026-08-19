from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)
# Calculating Mean Squared Error
def calculate_mse(y_ture,y_pred):
    return mean_squared_error(y_ture,y_pred)

# Calculating Mean Absolute Error
def calculate_mae(y_ture,y_pred):
    return mean_absolute_error(y_ture,y_pred)

# Calculating R2 Score
def calculate_r2(y_ture,y_pred):
    return r2_score(y_ture,y_pred)
