# %%
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
import numpy as np
import pandas as pd

from utils import compute_metrics

# %%
# Load the dataset
df = pd.read_csv("../data/cdcNormalDiabetic.csv")
print(df.shape)
print(df.columns)

df.head()

# %%
df.info()

# %%
df['Label'].value_counts()

# %%
feature_cols = [col for col in df.columns if col != 'Label']
print(f"Number of features: {len(feature_cols)}")

# %%
X = df[feature_cols]
y = df['Label']

X.shape, y.shape

# %%
# Split the data into training and test sets
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)
X_train_val.shape, y_train_val.shape, X_test.shape, y_test.shape

# %%
n_splits = 5
random_state = 42
early_stopping_rounds = 50

kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

# ---- Track scores ----
fold_accuracies, fold_precisions, fold_recalls, fold_f1s = [], [], [], []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_val)):
    print(f"\n--- Fold {fold + 1} ---")

    X_train, X_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    # Pool allows CatBoost to optimize for categorical features and speed
    train_pool = Pool(data=X_train, label=y_train)
    val_pool = Pool(data=X_val, label=y_val)

    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
        verbose=0,
        random_seed=random_state,
        eval_metric='Accuracy'
    )

    model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=early_stopping_rounds)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='binary')
    recall = recall_score(y_val, y_pred, average='binary')
    f1 = f1_score(y_val, y_pred, average='binary')
    fold_accuracies.append(acc)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1s.append(f1)

    print(f"Fold {fold + 1} Accuracy: {acc:.4f}")
    print(f"Fold {fold + 1} Precision: {precision:.4f}")
    print(f"Fold {fold + 1} Recall: {recall:.4f}")
    print(f"Fold {fold + 1} F1 Score: {f1:.4f}")

# ---- Summary ----
print(f"\nAverage Accuracy: {np.mean(fold_accuracies):.4f} ± {np.std(fold_accuracies):.4f}")
print(f"Average Precision: {np.mean(fold_precisions):.4f} ± {np.std(fold_precisions):.4f}")
print(f"Average Recall: {np.mean(fold_recalls):.4f} ± {np.std(fold_recalls):.4f}")
print(f"Average F1 Score: {np.mean(fold_f1s):.4f} ± {np.std(fold_f1s):.4f}")

# %%
# Evaluate on the test set
y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))

# %%
# Compute metrics for the test set
avg_options = ['micro', 'macro', 'weighted', 'binary']

results = [compute_metrics(y_test, y_pred, avg) for avg in avg_options]
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
