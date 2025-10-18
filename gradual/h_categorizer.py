import numpy as np


def build_att(A, R):
    """Builds a dictionary listing attackers for each argument."""
    att_list = {a: [] for a in A}
    for att, target in R:
        if target in att_list:
            att_list[target].append(att)
        else:
            att_list[target] = [att]
    return att_list


def h_categorizer(A, R, w, max_iter, epsi=1e-4):
    """Computes the h-Categorizer gradual semantics for a given framework (A, R) and weights."""
    attackers = build_att(A, R)
    hc = {a: w[a] for a in A}

    for _ in range(max_iter):
        new_hc = {}
        for a in A:
            sum_attackers = sum(hc[b] for b in attackers[a])
            new_hc[a] = w[a] / (1 + sum_attackers)
        diff = max(abs(new_hc[a] - hc[a]) for a in A)
        hc = new_hc
        if diff < epsi:
            break

    return hc
