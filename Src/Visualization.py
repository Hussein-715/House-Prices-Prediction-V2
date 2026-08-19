import matplotlib.pyplot as plt


def plot_feature_target(df,feature_name,target_name):
    plt.figure(figsize=(8, 5))
    plt.scatter(df[feature_name],df[target_name])
    plt.title(f"{feature_name} vs {target_name}")
    plt.xlabel(feature_name)
    plt.ylabel(target_name)
    plt.grid(True)
    plt.show()

def plot_predictions(y_true,y_predicted):
    min_value = min(y_true.min(),y_predicted.min())
    max_value = max(y_true.max(),y_predicted.max())
    x_coordinates = [min_value,max_value]
    y_coordinates = x_coordinates.copy()
    plt.figure(figsize=(8, 5))
    plt.plot(x_coordinates,y_coordinates,label="Perfect Predictions")
    plt.scatter(y_true,y_predicted,label="Predictions")
    plt.title("Predicted vs Actual Values")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_distribution(df,column_name):
    plt.figure(figsize=(8, 5))
    plt.hist(df[column_name])
    plt.title(f"{column_name} Distribution")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()