from sklearn.linear_model import LinearRegression

# Train the Linear Regression Model
def train_model(X_train,y_train):
    model = LinearRegression()
    model.fit(X_train,y_train)
    return model

