import pandas as pd

# Missing numerical values are replaced by the column's mean
# Missing categorial values are replaced by the column's Mode
# Rows with missing target value will be removed

def handle_missing_values(df,column_target):
    df = df.copy()
    df = df.dropna(subset=[column_target])
    for col_name in df.columns:
        if not df[col_name].isna().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col_name]):
            mean = df[col_name].mean()
            df[col_name] = df[col_name].fillna(mean)
        else:
            mode = df[col_name].mode()
            df[col_name] = df[col_name].fillna(mode.iloc[0])
    return df


# The target column (median_house_value) is extracted as the prediction target,
# while the remaining numerical features are used as the model's input.
# Categorical features are excluded at this stage to ensure that the feature matrix contains only numerical values
# compatible with the Linear Regression algorithm.

def split_features_target(df,target_column):
    y = df[target_column].to_numpy()
    df_features = df.drop(target_column,axis=1)
    columns_to_drop = [ ]
    for col_name in df_features.columns:
        if not pd.api.types.is_numeric_dtype(df_features[col_name]):
            columns_to_drop.append(col_name)
    df_features = df_features.drop(columns=columns_to_drop)
    X = df_features.to_numpy()
    return X,y


