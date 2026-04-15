import torch

def ROSA(x):
    """
    Rapid Online Suffix Automaton (ROSA)
    space = O(n), time = adaptive, typical O(n), worst-case O(n^2)
    """
    n = len(x)
    y = [-1] * n
    s = 2 * n + 1
    b = [None] * s
    c = [-1] * s
    d = [0] * s
    e = [-1] * s
    b[0] = {}
    g = 0
    z = 1
    
    for i, t in enumerate(x):
        r = z
        z += 1
        b[r] = {}
        d[r] = d[g] + 1
        p = g
        while p != -1 and t not in b[p]:
            b[p][t] = r
            p = c[p]
        if p == -1:
            c[r] = 0
        else:
            q = b[p][t]
            if d[p] + 1 == d[q]:
                c[r] = q
            else:
                u = z
                z += 1
                b[u] = b[q].copy()
                d[u] = d[p] + 1
                c[u] = c[q]
                e[u] = e[q]
                while p != -1 and b[p][t] == q:
                    b[p][t] = u
                    p = c[p]
                c[q] = c[r] = u
                
        v = g = r
        a = -1
        while v != -1:
            if d[v] > 0 and e[v] >= 0:
                a = x[e[v] + 1]
                break
            v = c[v]
            
        y[i] = a
        v = g
        while v != -1 and e[v] < i:
            e[v] = i
            v = c[v]
            
    return y
