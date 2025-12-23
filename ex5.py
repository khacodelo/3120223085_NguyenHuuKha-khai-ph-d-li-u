# Naive Bayes (categorical) for Play Tennis-like dataset
# Predict: Outlook=Sunny, Temp=Cool, Humidity=High, Windy=True

import pandas as pd
from math import prod

# ---------------------------
# 1) Data (IDs 1..14 are training)
# ---------------------------
data = [
    (1,  "Sunny",    "Hot",  "High",   False, "No"),
    (2,  "Sunny",    "Hot",  "High",   True,  "No"),
    (3,  "Overcast", "Hot",  "High",   False, "Yes"),
    (4,  "Rainy",    "Mild", "High",   False, "Yes"),
    (5,  "Rainy",    "Cool", "Normal", False, "Yes"),
    (6,  "Rainy",    "Cool", "Normal", True,  "No"),
    (7,  "Overcast", "Cool", "Normal", True,  "Yes"),
    (8,  "Sunny",    "Mild", "High",   False, "No"),
    (9,  "Sunny",    "Cool", "Normal", False, "Yes"),
    (10, "Rainy",    "Mild", "Normal", False, "Yes"),
    (11, "Sunny",    "Mild", "Normal", True,  "Yes"),
    (12, "Overcast", "Mild", "High",   True,  "Yes"),
    (13, "Overcast", "Hot",  "Normal", False, "Yes"),
    (14, "Rainy",    "Mild", "High",   True,  "No"),
]

df = pd.DataFrame(data, columns=["ID", "Outlook", "Temp", "Humidity", "Windy", "Play"])
features = ["Outlook", "Temp", "Humidity", "Windy"]
target = "Play"

# Mẫu cần dự đoán (ID=15)
x = {"Outlook": "Sunny", "Temp": "Cool", "Humidity": "High", "Windy": True}

# ---------------------------
# 2) Train counts
# ---------------------------
class_counts = df[target].value_counts().to_dict()
N = len(df)
classes = sorted(class_counts.keys())

# Số giá trị khác nhau (domain size) cho từng thuộc tính
domain_sizes = {f: df[f].nunique() for f in features}

def cond_prob(feature, value, cls, laplace=1):
    """
    P(feature=value | class=cls) with Laplace smoothing
    = (count(value in cls) + laplace) / (count(cls) + laplace * V)
    where V is number of distinct values of the feature.
    """
    df_c = df[df[target] == cls]
    count_value_in_c = (df_c[feature] == value).sum()
    V = domain_sizes[feature]
    return (count_value_in_c + laplace) / (len(df_c) + laplace * V)

def prior_prob(cls):
    return class_counts[cls] / N

# ---------------------------
# 3) Compute posterior scores (unnormalized)
# ---------------------------
scores = {}
details = {}

for cls in classes:
    prior = prior_prob(cls)
    conds = []
    for f in features:
        p = cond_prob(f, x[f], cls, laplace=1)  # Laplace=1
        conds.append(p)
    score = prior * prod(conds)
    scores[cls] = score
    details[cls] = {"prior": prior, "conds": dict(zip(features, conds))}

# Normalize to get comparable probabilities
total = sum(scores.values())
posteriors = {cls: scores[cls] / total for cls in classes}

# ---------------------------
# 4) Print result nicely
# ---------------------------
print("=== Input x (ID=15) ===")
print(x)

print("\n=== Class counts / Priors ===")
for cls in classes:
    print(f"{cls:>3}: count={class_counts[cls]}, prior={details[cls]['prior']:.6f}")

print("\n=== Conditional probabilities with Laplace smoothing (alpha=1) ===")
for cls in classes:
    print(f"\nClass = {cls}")
    for f in features:
        print(f"  P({f}={x[f]} | {cls}) = {details[cls]['conds'][f]:.6f}")

print("\n=== Unnormalized scores: prior * product(conds) ===")
for cls in classes:
    print(f"{cls:>3}: score = {scores[cls]:.12f}")

print("\n=== Normalized posteriors ===")
for cls in classes:
    print(f"{cls:>3}: P({cls} | x) = {posteriors[cls]:.6f}")

pred = max(posteriors, key=posteriors.get)
print("\n>>> Prediction for ID=15:", pred)
