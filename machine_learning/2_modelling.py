import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

val_test = pd.read_csv("data/processed/processed_test_val.csv")
train = pd.read_csv("data/processed/processed_train.csv")

val, test = train_test_split(
    val_test,
    test_size = 0.5,
    random_state = 42,
    stratify = val_test["driver_class"]
)

X_train = train.drop(columns = ["driver_class"])
y_train = train["driver_class"]

X_val = val.drop(columns = ["driver_class"])
y_val = val["driver_class"]

X_test = test.drop(columns = ["driver_class"])

model = RandomForestClassifier(
    n_estimators = 100
    max_depth = 5,
    random_state = 42
)
model.fit(X_train, y_train)

#[:,1] => I only keep B class probs
val_probs = model.predict_proba(X_val)[:, 1]

best_threshold = 0.5
best_score = 0.0

for thresh in np.linspace(0, 1, 101):
    val_preds = (val_probs >= thresh).astype(int)
    precision = precision_score(y_val, val_preds, pos_label = 1)
    recall = recall_score(y_val, val_preds, pos_label = 1)

    if recall > best_score and precision >= 0.5:
        best_score = recall
        best_threshold = thresh

print("Best threshold found is: ", best_threshold)

val_preds = (val_probs >= best_threshold).astype(int)
final_precision = precision_score(y_val, val_preds, pos_label = 1)
final_recall = recall_score(y_val, val_preds, pos_label = 1)
print("Validation precision: ", final_precision)
print("Validation recall: ", final_recall)

test_probs = model.predict_proba(X_test)[:,1]
test_preds = (test_probs >= best_threshold).astype(int)

pd.Dataframe({"driver_class": test_preds}).to_csv("predictions/prediction.csv", index = False)