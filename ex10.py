from itertools import combinations

def pretty(itemset):
    return "{" + ",".join(map(str, itemset)) + "}"

def sort_itemsets(sets_list):
    return sorted([tuple(sorted(s)) for s in sets_list])

# Frequent 3-itemsets (L3)
L3 = {
    (1, 2, 3),
    (1, 2, 4),
    (1, 2, 5),
    (1, 3, 4),
    (1, 3, 5),
    (2, 3, 4),
    (2, 3, 5),
    (3, 4, 5),
}

# All items (F1)
F1 = {1, 2, 3, 4, 5}

# (a) Fk-1 x F1 merging: add one new item into each 3-itemset
C4_a = set()
for s in L3:
    s_set = set(s)
    for x in F1:
        if x not in s_set:
            C4_a.add(tuple(sorted(s_set | {x})))

# (b) Apriori join step: join two 3-itemsets with same first (k-2)=2 items
L3_sorted = sorted(L3)  # tuples are already sorted
C4_b = set()
k = 4
prefix_len = k - 2  # 2

for i in range(len(L3_sorted)):
    for j in range(i + 1, len(L3_sorted)):
        a = L3_sorted[i]
        b = L3_sorted[j]
        if a[:prefix_len] == b[:prefix_len]:
            candidate = tuple(sorted(set(a) | set(b)))
            if len(candidate) == 4:
                C4_b.add(candidate)

# (c) Apriori prune step: keep candidate if all 3-subsets are in L3
L3_set = set(L3)
C4_survive = set()

for c in C4_b:
    all_3_subsets = list(combinations(c, 3))
    if all(sub in L3_set for sub in all_3_subsets):
        C4_survive.add(c)

# Print results
print("L3 (Frequent 3-itemsets):")
for s in sort_itemsets(L3):
    print(" ", pretty(s))

print("\n(a) Candidates C4 from Fk-1 x F1 merging:")
for s in sort_itemsets(C4_a):
    print(" ", pretty(s))

print("\n(b) Candidates C4 from Apriori join:")
for s in sort_itemsets(C4_b):
    print(" ", pretty(s))

print("\n(c) Candidates that survive Apriori pruning:")
for s in sort_itemsets(C4_survive):
    print(" ", pretty(s))
