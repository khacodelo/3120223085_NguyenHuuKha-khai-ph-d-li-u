import math
from collections import Counter

# ======================
# Dataset
# ======================
# Each instance: (a1, a2, class)
data = [
    ('T', 'T', '+'),
    ('T', 'T', '+'),
    ('T', 'F', '-'),
    ('F', 'F', '+'),
    ('F', 'T', '-'),
    ('F', 'T', '-'),
    ('F', 'F', '-'),
    ('T', 'F', '+'),
    ('F', 'T', '-')
]

# ======================
# Entropy
# ======================
def entropy(labels):
    total = len(labels)
    counter = Counter(labels)
    ent = 0
    for count in counter.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent

# ======================
# Information Gain
# ======================
def information_gain(data, attr_index):
    total_entropy = entropy([row[2] for row in data])
    total = len(data)

    subsets = {}
    for row in data:
        key = row[attr_index]
        subsets.setdefault(key, []).append(row)

    weighted_entropy = 0
    for subset in subsets.values():
        weighted_entropy += (len(subset) / total) * entropy([row[2] for row in subset])

    return total_entropy - weighted_entropy

# ======================
# Gini Index
# ======================
def gini(labels):
    total = len(labels)
    counter = Counter(labels)
    g = 1
    for count in counter.values():
        p = count / total
        g -= p ** 2
    return g

def gini_split(data, attr_index):
    total = len(data)
    subsets = {}

    for row in data:
        key = row[attr_index]
        subsets.setdefault(key, []).append(row)

    gini_after = 0
    for subset in subsets.values():
        gini_after += (len(subset) / total) * gini([row[2] for row in subset])

    return gini_after

# ======================
# MAIN
# ======================
labels = [row[2] for row in data]

print("Entropy(S) =", round(entropy(labels), 4))

print("\nInformation Gain:")
print("IG(a1) =", round(information_gain(data, 0), 4))
print("IG(a2) =", round(information_gain(data, 1), 4))

print("\nGini Index:")
print("Gini(a1) =", round(gini_split(data, 0), 4))
print("Gini(a2) =", round(gini_split(data, 1), 4))

best_gini = "a1" if gini_split(data, 0) < gini_split(data, 1) else "a2"
print("\nBest split according to Gini:", best_gini)
