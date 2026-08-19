from Data_Loader import load_data , inspect_data
from Preprocessing import handle_missing_values , split_features_target
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from Model import train_model
from Evaluation import calculate_mae,calculate_mse,calculate_r2
from Visualization import plot_feature_target,plot_predictions,plot_distribution


# Loading a Dataset From a CSV File
df = load_data("Data/California_Housing.csv")

# Printing the DataFrame information
inspect_data(df)
print("\n")

# Handling the missing values in our DataFrame
df = handle_missing_values(df,"median_house_value")

# Check our DataFrame after handling it
inspect_data(df)
print("\n")

# Extracting X feature matrix and y target vector from our DataFrame
X , y = split_features_target(df,"median_house_value")

# Check that the categorial columns are removed
print(X)
print("\n")

# Split our Data to Training set and Testing set
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Scale our training and testing features X
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train our LinearRegression Model
model = train_model(X_train,y_train)

# Get our Test Predictions from our X testing values
y_test_pred = model.predict(X_test)

# Printing DataSet Shapes
print("Training Samples : ",X_train.shape[0])
print("Testing Samples : ",X_test.shape[0])
print("Number Of Features : ",X_train.shape[1])
print("\n")

# Printing Model Parameters
print("w1,w2,...,wj : ",model.coef_)
print("b : ",model.intercept_)
print("\n")

# Evaluate our model predictions on testing set
mse = calculate_mse(y_test,y_test_pred)
print(f"Mean_Squared_Error : {mse}")
mae = calculate_mae(y_test,y_test_pred)
print(f"Mean_Absolute_Error : {mae}")
r2 = calculate_r2(y_test,y_test_pred)
print(f"R2_Score : {r2}")

# Visualize our Feature/Targets , Predictions/Targets and Feature Distribution
plot_feature_target(df,"median_income","median_house_value")
plot_predictions(y_test,y_test_pred)
plot_distribution(df,"median_house_value")
