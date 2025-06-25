# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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
# Split the data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
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
    learning_rate=0.01,
    max_depth=10,
    verbose=0,
    random_seed=random_state,
    # auto_class_weights='SqrtBalanced',  # better than 'Balanced'
    # class_weights=[1, 3],  # Adjust class weights for imbalance
    eval_metric="AUC",
)

model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)

y_prob = model.predict_proba(X_val)[:, 1]
thresholds = np.linspace(0, 1, 100)
f1_scores = [f1_score(y_val, y_prob > t) for t in thresholds]

best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]
print(f"Best threshold = {best_threshold:.2f}, F1 = {f1_scores[best_idx]:.4f}")

# Plot Threshold vs F1 Score
plt.plot(thresholds, f1_scores)
plt.xlabel("Threshold")
plt.ylabel("F1 Score")
plt.title("Threshold Optimization")
plt.grid(True)
plt.show()
