from collections import Counter, defaultdict

# ====== 1) Nhập dữ liệu đúng theo bảng ======
data = [
    # (A, B, C, Class)
    (0, 0, 0, '+'),  # 1
    (0, 0, 1, '-'),  # 2
    (0, 1, 1, '-'),  # 3
    (0, 1, 1, '-'),  # 4
    (0, 0, 1, '+'),  # 5
    (1, 0, 1, '+'),  # 6
    (1, 0, 1, '-'),  # 7
    (1, 0, 1, '-'),  # 8
    (1, 1, 1, '+'),  # 9
    (1, 0, 1, '+'),  # 10
]

features = ['A', 'B', 'C']

# ====== 2) Đếm prior ======
class_counts = Counter(cls for *_, cls in data)
N = len(data)
priors = {cls: class_counts[cls] / N for cls in class_counts}

print("=== PRIOR (P(Class)) ===")
for cls in sorted(priors):
    print(f"P({cls}) = {class_counts[cls]}/{N} = {priors[cls]:.4f}")
print()

# ====== 3) Bảng đếm theo lớp cho từng feature ======
# counts[cls][feature][value] = count
counts = {cls: {f: Counter() for f in features} for cls in class_counts}

for A, B, C, cls in data:
    vals = {'A': A, 'B': B, 'C': C}
    for f in features:
        counts[cls][f][vals[f]] += 1

print("=== BẢNG ĐẾM (theo lớp) ===")
for cls in sorted(counts):
    print(f"Class {cls} (N={class_counts[cls]}):")
    for f in features:
        c1 = counts[cls][f][1]
        c0 = counts[cls][f][0]
        print(f"  {f}: count(1|{cls})={c1}, count(0|{cls})={c0}")
    print()

# ====== 4) Tính conditional probabilities (không smoothing) ======
cond = {cls: {f: {} for f in features} for cls in class_counts}

for cls in class_counts:
    Nc = class_counts[cls]
    for f in features:
        for v in [0, 1]:
            cond[cls][f][v] = counts[cls][f][v] / Nc

print("=== CONDITIONAL PROB (không smoothing) ===")
for cls in sorted(cond):
    for f in features:
        print(f"P({f}=1|{cls}) = {counts[cls][f][1]}/{class_counts[cls]} = {cond[cls][f][1]:.4f}")
    print()

# ====== 5) Dự đoán mẫu test ======
x = {'A': 0, 'B': 1, 'C': 0}

def score_no_smoothing(cls):
    s = priors[cls]
    for f in features:
        s *= cond[cls][f][x[f]]
    return s

scores = {cls: score_no_smoothing(cls) for cls in class_counts}
print("=== DỰ ĐOÁN (không smoothing) ===")
for cls in sorted(scores):
    print(f"Score({cls}) = {scores[cls]:.6f}")
pred = max(scores, key=scores.get)
print("=> Predict:", pred)
print()

# ====== 6) (Tuỳ chọn) Laplace smoothing để tránh xác suất 0 ======
# Vì mỗi feature nhị phân {0,1} => +2 ở mẫu số
def score_laplace(cls):
    Nc = class_counts[cls]
    s = priors[cls]  # giữ nguyên prior
    for f in features:
        count_v = counts[cls][f][x[f]]
        p = (count_v + 1) / (Nc + 2)
        s *= p
    return s

scores_lap = {cls: score_laplace(cls) for cls in class_counts}
print("=== DỰ ĐOÁN (Laplace smoothing) ===")
for cls in sorted(scores_lap):
    print(f"Score({cls}) = {scores_lap[cls]:.6f}")
pred_lap = max(scores_lap, key=scores_lap.get)
print("=> Predict (Laplace):", pred_lap)
