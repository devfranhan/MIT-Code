def vocab_no_sw(texts, remove=False):
    sw = set()
    if remove:
        with open("stopwords.txt", "r") as f:
            sw = set(f.read().split())

    v = {}
    for txt in texts:
        words = txt.lower().split()
        for w in words:
            if w not in sw and w not in v:
                v[w] = len(v)

    return v

# load stopwords as lowercase only to avoid mismatch?
