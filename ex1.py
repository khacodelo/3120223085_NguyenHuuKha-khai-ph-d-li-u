# KNN 1D - classify z = 5.0 with k = 1, 3, 5, 9 (majority vote)

def knn_predict_1d(x_train, y_train, z, k):
    # Tính khoảng cách |x - z|
    distances = []
    for x, y in zip(x_train, y_train):
        d = abs(x - z)
        distances.append((d, x, y))  # (distance, x_value, label)

    # Sắp xếp tăng dần theo khoảng cách
    distances.sort(key=lambda t: t[0])

    # Lấy k láng giềng gần nhất
    neighbors = distances[:k]

    # Đếm phiếu
    count_plus = sum(1 for _, _, lab in neighbors if lab == '+')
    count_minus = sum(1 for _, _, lab in neighbors if lab == '-')

    # Dự đoán theo majority vote
    if count_plus > count_minus:
        pred = '+'
    elif count_minus > count_plus:
        pred = '-'
    else:
        pred = 'Hòa phiếu (tie)'  # trường hợp đặc biệt nếu bằng nhau

    return pred, neighbors, count_plus, count_minus


def main():
    # Dataset theo hình
    x_train = [0.5, 3.0, 4.5, 4.6, 4.9, 5.2, 5.3, 5.5, 7.0, 9.5]
    y_train = ['-', '-', '+', '+', '+', '-', '-', '+', '-', '-']

    z = 5.0
    ks = [1, 3, 5, 9]

    print("=== KNN 1D Classification ===")
    print("Train data:")
    for x, y in zip(x_train, y_train):
        print(f"  x={x:>4}  ->  y={y}")
    print(f"\nPoint to classify: z = {z}\n")

    for k in ks:
        pred, neighbors, c_plus, c_minus = knn_predict_1d(x_train, y_train, z, k)

        print(f"--- k = {k} ---")
        print("Nearest neighbors (distance, x, label):")
        for d, x, lab in neighbors:
            print(f"  d={d:.2f}   x={x:>4}   y={lab}")

        print(f"Vote: '+' = {c_plus}, '-' = {c_minus}")
        print(f"Prediction for z={z} => {pred}\n")


if __name__ == "__main__":
    main()
