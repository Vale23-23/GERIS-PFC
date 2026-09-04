import numpy as np
from sklearn.tree import DecisionTreeClassifier,export_text,plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,recall_score,f1_score
import matplotlib.pyplot as plt
import random

X = np.array([
    [25.0, 40.0],
    [30.0, 30.0],
    [20.0, 70.0],
    [35.0, 20.0],
    [28.0, 50.0],
])

y = np.array([
    0,
    1,
    0,
    1,
    1
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


tree = DecisionTreeClassifier(random_state=42, max_depth=5)

print("Training...")
tree.fit(X_train,y_train)
print("Predicting...")
predictions = tree.predict(X_test,y_test)
print("Results:")
print(accuracy_score(y_test,predictions))
print(recall_score(y_test,predictions))
print(f1_score(y_test,predictions))

'''plt.figure(figsize=(12, 8))

plot_tree(
    tree,
    feature_names=["temperature", "humidity"],
    class_names=["No fire", "Fire"],
    filled=True
)

plt.show()'''


import goes
import pandas as pd

date = "2025-11-17 18:00"
b07,b14,fire = goes.get_goes_data_from_timestamp(pd.to_datetime(date))

# Create one figure with 3 images side-by-side
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# B07
vmin, vmax = np.percentile(b07, [2, 98])
axes[0].imshow(b07, cmap="inferno", vmin=vmin, vmax=vmax)
axes[0].set_title("ABI B07")
axes[0].axis("off")

# B14
vmin, vmax = np.percentile(b14, [2, 98])
axes[1].imshow(b14, cmap="RdYlBu_r", vmin=vmin, vmax=vmax)
axes[1].set_title("ABI B14")
axes[1].axis("off")

# L2 Fire
axes[2].imshow(fire, cmap="hot")
axes[2].set_title("L2 Fire Product")
axes[2].axis("off")

plt.tight_layout()
plt.show()