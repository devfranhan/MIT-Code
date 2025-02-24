def vocab(texts):
    v = {}
    for txt in texts:
        words = txt.lower().split()
        for w in words:
            if w not in v:
                v[w] = len(v)
    return v


# sorting might improve lookup performance
