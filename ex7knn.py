import math

# ===== 1. Dữ liệu =====
points = {
    "A1": (2, 10),
    "A2": (2, 5),
    "A3": (8, 4),
    "A4": (5, 8),
    "A5": (7, 5),
    "A6": (6, 4),
    "A7": (1, 2),
    "A8": (4, 9)
}

# ===== 2. Tâm ban đầu =====
centroids = {
    "C1": points["A1"],
    "C2": points["A2"],
    "C3": points["A3"]
}

# ===== 3. Hàm tính khoảng cách Euclid =====
def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ===== 4. Hàm gán cụm =====
def assign_clusters(points, centroids):
    clusters = {k: [] for k in centroids}
    for name, point in points.items():
        nearest = min(centroids,
                      key=lambda c: distance(point, centroids[c]))
        clusters[nearest].append(name)
    return clusters

# ===== 5. Hàm tính tâm mới =====
def update_centroids(points, clusters):
    new_centroids = {}
    for cluster, names in clusters.items():
        x_avg = sum(points[n][0] for n in names) / len(names)
        y_avg = sum(points[n][1] for n in names) / len(names)
        new_centroids[cluster] = (x_avg, y_avg)
    return new_centroids

# ===== 6. Chạy thuật toán K-means =====
iteration = 1
while True:
    print(f"\n--- Lặp {iteration} ---")

    clusters = assign_clusters(points, centroids)
    for c, pts in clusters.items():
        print(f"{c}: {pts}")

    new_centroids = update_centroids(points, clusters)

    print("Tâm mới:")
    for c in new_centroids:
        print(f"{c}: {new_centroids[c]}")

    # kiểm tra hội tụ
    if new_centroids == centroids:
        print("\n👉 Thuật toán hội tụ. Kết thúc.")
        break

    centroids = new_centroids
    iteration += 1
