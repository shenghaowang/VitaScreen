# %%
import pandas as pd
from catboost import CatBoostClassifier, Pool
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# %%
# Load the dataset
df = pd.read_csv("../data/cdcNormalDiabetic.csv")
print(df.shape)
print(df.columns)

# %%
df["Label"].value_counts()

# %%
feature_cols = [col for col in df.columns if col != "Label"]
print(f"Number of features: {len(feature_cols)}")

# %%
X = df[feature_cols]
y = df["Label"]

X.shape, y.shape

# %%

# Resampling using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
X_resampled.shape, y_resampled.shape

# %%
# Split the data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.3, random_state=42
)
X_train_val.shape, y_train_val.shape, X_test.shape, y_test.shape

# %%
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)
X_train.shape, y_train.shape, X_val.shape, y_val.shape

# %%
train_pool = Pool(data=X_train, label=y_train)
val_pool = Pool(data=X_val, label=y_val)

random_state = 42
early_stopping_rounds = 50

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=0,
    random_seed=random_state,
    eval_metric="Accuracy",
)

model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)

y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, average="binary")
recall = recall_score(y_val, y_pred, average="binary")
f1 = f1_score(y_val, y_pred, average="binary")


print(f"Validation Accuracy: {acc:.4f}")
print(f"Validation Precision: {precision:.4f}")
print(f"Validation Recall: {recall:.4f}")
print(f"Validation F1 Score: {f1:.4f}")

# Validation Accuracy: 0.9072
# Validation Precision: 0.9620
# Validation Recall: 0.8484
# Validation F1 Score: 0.9016

# %%
# Evaluate on the test set
y_pred = model.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
test_precision = precision_score(y_test, y_pred, average="binary")
test_recall = recall_score(y_test, y_pred, average="binary")
test_f1 = f1_score(y_test, y_pred, average="binary")
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Test Accuracy: 0.9089
# Test Precision: 0.9638
# Test Recall: 0.8493
# Test F1 Score: 0.9029
