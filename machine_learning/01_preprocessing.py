import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

#OneHotEncoding is suitable here since there are only two categorical features with a small number of categories.


result = pd.read_csv("data/processed/collected.csv")

#Split the dataset into 70% training and 30% testing sets
    #I use stratify to preserve the original class distribution (A/B), which is important due to class imbalance.
    #random_state=42 ensures reproducibility of the split
train, test_val = train_test_split(
    result,
    test_size = 0.3,
    random_state = 42,
    stratify = result["driver_class"]
)

#Fill Nan_Value in age cols with train data to avoid dataleakage 
avg_age = round(train["age"].mean(), 0)
train["age"] = train["age"].fillna(avg_age).astype(int)
test_val["age"] = test_val["age"].fillna(avg_age).astype(int)

#Categorical encoding (OneHotEncoder) for "car_model" and "second_language"
cat_cols = [
    "car_model",
    "second_language"
]

#handle_unknown prevents errors if new categories appear in the test dataset that were not present in the training dataset
#sparse=False returns a dense array, which is easier to convert to a pandas DataFrame
onehotenc = OneHotEncoder(handle_unknown = "ignore", sparse_output = False) 

#Fit only on train dataset
onehotenc.fit(train[cat_cols])

#Transform train and test
train_encoded = (
    pd.DataFrame(
        onehotenc.transform(train[cat_cols]),
        columns = onehotenc.get_feature_names_out(cat_cols),
        index = train.index
    )
)

test_val_encoded = (
    pd.DataFrame(
        onehotenc.transform(test_val[cat_cols]),
        columns = onehotenc.get_feature_names_out(cat_cols),
        index = test_val.index
    )
)

#Drop the original categorical columns to keep only the encoded features
train = pd.concat([train.drop(columns=cat_cols), train_encoded], axis=1)
test_val = pd.concat([test_val.drop(columns=cat_cols), test_val_encoded], axis=1)

#Standardize tips for baseline model with LogisticRegression 
scaler = StandardScaler()
train["net_worth_of_tips"] = scaler.fit_transform(train[["net_worth_of_tips"]])
test_val["net_worth_of_tips"] = scaler.transform(test_val[["net_worth_of_tips"]])

#Mapping: class A = 0 and class B = 1
mapping = ({"A class": 0, "B class": 1})
train["driver_class"] = train["driver_class"].map(mapping)
test_val["driver_class"] = test_val["driver_class"].map(mapping) 

#Save dataset
train.to_csv("data/processed/processed_train.csv", index = False)
test_val.to_csv("data/processed/processed_test_val.csv", index = False)

