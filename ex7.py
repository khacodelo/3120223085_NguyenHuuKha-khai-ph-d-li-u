from collections import Counter, defaultdict

# ===== 1) Dữ liệu (đúng theo bảng 20 mẫu) =====
data = [
    # (CustomerID, Gender, CarType, ShirtSize, Class)
    (1,  "M", "Family", "Small",       "C0"),
    (2,  "M", "Sports", "Medium",      "C0"),
    (3,  "M", "Sports", "Medium",      "C0"),
    (4,  "M", "Sports", "Large",       "C0"),
    (5,  "M", "Sports", "Extra Large", "C0"),
    (6,  "M", "Sports", "Extra Large", "C0"),
    (7,  "F", "Sports", "Small",       "C0"),
    (8,  "F", "Sports", "Small",       "C0"),
    (9,  "F", "Sports", "Medium",      "C0"),
    (10, "F", "Luxury", "Large",       "C0"),
    (11, "M", "Family", "Large",       "C1"),
    (12, "M", "Family", "Extra Large", "C1"),
    (13, "M", "Family", "Medium",      "C1"),
    (14, "M", "Luxury", "Extra Large", "C1"),
    (15, "F", "Luxury", "Small",       "C1"),
    (16, "F", "Luxury", "Small",       "C1"),
    (17, "F", "Luxury", "Medium",      "C1"),
    (18, "F", "Luxury", "Medium",      "C1"),
    (19, "F", "Luxury", "Medium",      "C1"),
    (20, "F", "Luxury", "Large",       "C1"),
]

ATTRS = {
    "CustomerID": 0,
    "Gender": 1,
    "CarType": 2,
    "ShirtSize": 3,
}

# ===== 2) Hàm Gini =====
def gini(labels):
    """
    Gini(t) = 1 - sum_k (p_k^2)
    """
    n = len(labels)
    if n == 0:
        return 0.0
    cnt = Counter(labels)
    return 1.0 - sum((v / n) ** 2 for v in cnt.values())

def gini_split(data, attr_index):
    """
    Multiway split theo thuộc tính attr_index:
    Gini_split(A) = sum_v (|D_v|/|D|) * Gini(D_v)
    """
    groups = defaultdict(list)  # value -> list of class labels
    for row in data:
        groups[row[attr_index]].append(row[-1])

    n = len(data)
    weighted = 0.0
    detail = {}
    for val, labels in groups.items():
        w = len(labels) / n
        gv = gini(labels)
        weighted += w * gv
        detail[val] = {
            "size": len(labels),
            "class_count": dict(Counter(labels)),
            "gini": gv,
            "weight": w,
            "weighted_gini": w * gv,
        }
    return weighted, detail

# ===== 3) Tính toán & In kết quả =====
if __name__ == "__main__":
    # (a) Overall
    all_labels = [row[-1] for row in data]
    print("==== (a) Overall Gini ====")
    print("Class counts:", dict(Counter(all_labels)))
    print("Gini(overall) =", round(gini(all_labels), 6))
    print()

    results = {}

    # (b)(c)(d)(e) từng thuộc tính
    for attr_name, idx in ATTRS.items():
        split_gini, detail = gini_split(data, idx)
        results[attr_name] = split_gini

        print(f"==== Gini for attribute: {attr_name} (multiway split) ====")
        print("Gini_split =", round(split_gini, 6))
        print("Details by value:")
        # in theo thứ tự đẹp
        for val in sorted(detail.keys(), key=lambda x: str(x)):
            info = detail[val]
            print(f"  - {val}: size={info['size']}, class={info['class_count']}, "
                  f"gini={info['gini']:.6f}, weight={info['weight']:.3f}, "
                  f"w*gini={info['weighted_gini']:.6f}")
        print()

    # (f) chọn thuộc tính tốt nhất trong Gender, CarType, ShirtSize
    candidates = ["Gender", "CarType", "ShirtSize"]
    best = min(candidates, key=lambda k: results[k])

    print("==== (f) Best attribute among Gender, CarType, ShirtSize ====")
    for k in candidates:
        print(f"{k}: Gini_split = {results[k]:.6f}")
    print("=> Best =", best, "(min Gini_split)")
