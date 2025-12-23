from typing import List, Set, FrozenSet

def support_count(transactions: List[Set[str]], itemset: Set[str]) -> int:
    """Đếm số transaction chứa itemset"""
    return sum(1 for t in transactions if itemset.issubset(t))

def support(transactions: List[Set[str]], itemset: Set[str]) -> float:
    """Support = count(itemset) / total_transactions"""
    return support_count(transactions, itemset) / len(transactions)

def confidence(transactions: List[Set[str]], X: Set[str], Y: Set[str]) -> float:
    """Conf(X -> Y) = support_count(X ∪ Y) / support_count(X)"""
    union = X.union(Y)
    cnt_X = support_count(transactions, X)
    if cnt_X == 0:
        return 0.0
    return support_count(transactions, union) / cnt_X

def fmt_itemset(s: Set[str]) -> str:
    return "{" + ",".join(sorted(s)) + "}"

def main():
    # 10 transactions theo bảng
    transactions = [
        {"a", "d", "e"},        # 0001
        {"a", "b", "c", "e"},   # 0024
        {"a", "b", "d", "e"},   # 0012
        {"a", "c", "d", "e"},   # 0031
        {"b", "c", "e"},        # 0015
        {"b", "d", "e"},        # 0022
        {"c", "d"},             # 0029
        {"a", "b", "c"},        # 0040
        {"a", "d", "e"},        # 0033
        {"a", "b", "e"},        # 0038
    ]

    N = len(transactions)
    E = {"e"}
    BD = {"b", "d"}
    BDE = {"b", "d", "e"}

    # (a) Support cho {e}, {b,d}, {b,d,e}
    itemsets = [E, BD, BDE]
    print(f"Total transactions = {N}\n")

    print("=== (a) SUPPORT ===")
    for it in itemsets:
        cnt = support_count(transactions, it)
        sup = cnt / N
        print(f"SupportCount{fmt_itemset(it)} = {cnt}")
        print(f"Support{fmt_itemset(it)}      = {cnt}/{N} = {sup:.4f}\n")

    # (b) Confidence cho {b,d} -> {e} và {e} -> {b,d}
    print("=== (b) CONFIDENCE ===")
    conf_bd_to_e = confidence(transactions, BD, E)
    conf_e_to_bd = confidence(transactions, E, BD)

    # In kèm công thức số học
    cnt_bde = support_count(transactions, BDE)
    cnt_bd = support_count(transactions, BD)
    cnt_e = support_count(transactions, E)

    print(f"Conf{fmt_itemset(BD)} -> {fmt_itemset(E)} = "
          f"SupportCount{fmt_itemset(BDE)} / SupportCount{fmt_itemset(BD)} = "
          f"{cnt_bde}/{cnt_bd} = {conf_bd_to_e:.4f}")

    print(f"Conf{fmt_itemset(E)} -> {fmt_itemset(BD)} = "
          f"SupportCount{fmt_itemset(BDE)} / SupportCount{fmt_itemset(E)} = "
          f"{cnt_bde}/{cnt_e} = {conf_e_to_bd:.4f}")

if __name__ == "__main__":
    main()
