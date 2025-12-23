import math
from collections import Counter

# ===== 1) Data đúng theo bảng bạn gửi =====
data = [
    (4,3,'+'),
    (3,7,'+'),
    (7,4,'+'),
    (4,1,'+'),
    (6,5,'+'),
    (5,6,'+'),
    (3,7,'+'),
    (6,2,'+'),
    (4,6,'-'),
    (4,4,'-'),
    (5,8,'-'),
    (7,8,'-'),
    (7,6,'-'),
    (4,10,'-'),
    (9,7,'-'),
    (5,4,'-'),
    (8,5,'-'),
    (6,6,'-'),
    (7,4,'-'),
    (8,8,'-')
]

# ===== 2) Chọn test 2+ và 2- (đúng phần mình làm ở trên) =====
test_points = [
    ("T1", (4,3), '+'),
    ("T2", (7,4), '+'),
    ("T3", (7,8), '-'),
    ("T4", (7,6), '-')
]

# bỏ test ra khỏi train (so khớp theo tuple đầy đủ)
test_set_tuples = set((x1,x2,c) for _,(x1,x2),c in test_points)
train = [p for p in data if p not in test_set_tuples]

def euclid(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def knn_predict(train, query_xy, k):
    # tính khoảng cách đến tất cả train
    dlist = []
    for x1,x2,c in train:
        d = euclid((x1,x2), query_xy)
        dlist.append((d, (x1,x2), c))

    # sort theo khoảng cách
    dlist.sort(key=lambda t: t[0])

    # lấy k láng giềng
    neigh = dlist[:k]
    votes = Counter([c for _,_,c in neigh])

    pred = '+' if votes['+'] > votes['-'] else '-'
    return pred, neigh, votes

def confusion(y_true, y_pred):
    TP=FN=TN=FP=0
    for t,p in zip(y_true, y_pred):
        if t=='+' and p=='+': TP+=1
        elif t=='+' and p=='-': FN+=1
        elif t=='-' and p=='-': TN+=1
        elif t=='-' and p=='+': FP+=1
    return TP,FN,TN,FP

def print_neighbors(neigh, top=9):
    print("  Top neighbors:")
    for i,(d,xy,c) in enumerate(neigh[:top], start=1):
        print(f"   {i:>2}. {xy}  class={c}  d={d:.3f}")

for k in [5, 9]:
    print("="*60)
    print(f"K = {k}")
    y_true = []
    y_pred = []

    for name,xy,true_c in test_points:
        pred, neigh, votes = knn_predict(train, xy, k)
        y_true.append(true_c)
        y_pred.append(pred)

        print(f"\n{name}: query={xy}  TRUE={true_c}")
        print_neighbors(neigh, top=9)
        print(f"  Votes in k={k}: +={votes['+']}  -= {votes['-']}  => PRED={pred}")

    TP,FN,TN,FP = confusion(y_true, y_pred)
    acc = (TP+TN)/len(y_true)
    print("\nConfusion matrix (Positive='+'):")
    print(f"TP={TP}, FN={FN}, TN={TN}, FP={FP}")
    print(f"Accuracy = (TP+TN)/N = ({TP}+{TN})/{len(y_true)} = {acc*100:.2f}%")
S