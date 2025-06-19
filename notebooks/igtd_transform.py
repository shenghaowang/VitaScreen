# %%
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd

from utils import compute_metrics
from IGTD_Functions import min_max_transform, table_to_image, select_features_by_variation


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
# Create polynomial features
poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
X_poly = poly.fit_transform(df[feature_cols])
poly_feature_names = poly.get_feature_names_out(feature_cols)
df_poly = pd.DataFrame(X_poly, columns=poly_feature_names)

print(f"Number of polynomial features: {df_poly.shape[1]}")

# 252 = 4 * 7 * 9 = 14 * 18

# %%
# Select features with large variations across samples
X = df_poly[poly_feature_names]
id = select_features_by_variation(X, variation_measure='var', num=len(poly_feature_names))
X = X.iloc[:, id]
y = df['Label']

X.shape, y.shape

# %%
# Perform min-max transformation
X_norm = min_max_transform(X.values)
X_norm = pd.DataFrame(X_norm, columns=X.columns, index=X.index)

X_norm.shape

# %%
# Run the IGTD algorithm
nrows, ncols = 14, 18
save_image_size = 3
max_step, val_step = 30000, 300
fea_dist_method = 'Euclidean'
image_dist_method = 'Euclidean'
error = 'abs'
result_dir = 'IGTD/Test_2'
os.makedirs(name=result_dir, exist_ok=True)

start_time = time.time()
table_to_image(X_norm, [nrows, ncols], fea_dist_method, image_dist_method, save_image_size,
               max_step, val_step, result_dir, error)
end_time = time.time()

print(f"Training time: {(end_time - start_time):.2f} seconds")

