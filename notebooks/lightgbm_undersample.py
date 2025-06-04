# %%
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd

from utils import compute_metrics

# %%
# Load the dataset
df = pd.read_csv("../data/cdcNormalDiabetic.csv")
print(df.shape)
print(df.columns)

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
    X, y, test_size=0.2, random_state=42
)
X_train_val.shape, y_train_val.shape, X_test.shape, y_test.shape

# %%
# Apply Random Under Sampling to balance the training data
rus = RandomUnderSampler(random_state=42)
X_train_val, y_train_val = rus.fit_resample(X_train_val, y_train_val)
print(f"After undersampling: {y_train_val.value_counts()}")

# Split the balanced training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=42
)
X_train.shape, y_train.shape, X_val.shape, y_val.shape

# %%
random_state = 42
early_stopping_rounds = 50

model = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=10,
    random_state=random_state,
    objective='binary',
    metric='binary_logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='binary_logloss',
    callbacks=[
        early_stopping(stopping_rounds=early_stopping_rounds),
        log_evaluation(period=10)  # print every 10 rounds
    ],
)

y_pred = model.predict(X_val, num_iteration=model.best_iteration_)
val_metrics = compute_metrics(y_val, y_pred, avg_option='binary')

print(f"Validation Accuracy: {val_metrics['accuracy']:.4f}")
print(f"Validation Precision: {val_metrics['precision']:.4f}")
print(f"Validation Recall: {val_metrics['recall']:.4f}")
print(f"Validation F1 Score: {val_metrics['f1_score']:.4f}")

# Validation Accuracy: 0.7474
# Validation Precision: 0.7345
# Validation Recall: 0.7821
# Validation F1 Score: 0.7576

# %%
# Evaluate on the test set
y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
avg_options = ['micro', 'macro', 'weighted', 'binary']

results = [compute_metrics(y_test, y_pred, avg) for avg in avg_options]
results_df = pd.DataFrame(results)
results_df.to_csv("results.csv", index=False)
