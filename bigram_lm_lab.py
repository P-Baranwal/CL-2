import itertools
from collections import Counter

# Corpus: 4 tokenized sentences with <s> and </s>
sentences = [
    "<s> peter piper picked a peck of pickled peppers </s>",
    "<s> a peck of pickled peppers peter piper picked </s>",
    "<s> if peter piper picked a peck of pickled peppers </s>",
    "<s> where s the peck of pickled peppers peter piper picked </s>",
]

# Tokenize
tokenized = [s.split() for s in sentences]

# Build vocabulary
vocab = sorted(set(token for sent in tokenized for token in sent))
V = len(vocab)
print("Vocabulary:", vocab)
print(f"Vocabulary size V = {V}\n")

# Unigram counts (overall token frequency)
unigram_counts = Counter(token for sent in tokenized for token in sent)

# Bigram counts
bigram_counts = Counter()
for sent in tokenized:
    bigram_counts.update(zip(sent, sent[1:]))

total_bigram_tokens = sum(bigram_counts.values())  # N in Good-Turing

# ------------------------------------------------------------
# 1. Bigram probability table (MLE)
# P(w2|w1) = count(w1,w2) / count(w1)
# ------------------------------------------------------------
print("=" * 70)
print("1. BIGRAM PROBABILITY TABLE (MLE)")
print("=" * 70)
print(f"{'w1':<12} {'w2':<12} {'count':<7} {'P(w2|w1)':<10}")
print("-" * 70)
for (w1, w2), cnt in sorted(bigram_counts.items()):
    prob = cnt / unigram_counts[w1]
    print(f"{w1:<12} {w2:<12} {cnt:<7} {prob:.4f}")
print()

# ------------------------------------------------------------
# 2. Laplace (add-one) smoothing
# P_Laplace(w2|w1) = (count(w1,w2) + 1) / (count(w1) + V)
# ------------------------------------------------------------
print("=" * 70)
print("2. LAPLACE (ADD-ONE) SMOOTHING")
print("=" * 70)
print(f"Vocabulary size V = {V}")

# Pick an unseen bigram: </s> is never followed by anything, so (</s>, peter) is unseen
unseen_pair = ("</s>", "peter")
w1, w2 = unseen_pair
cnt_w1w2 = bigram_counts.get(unseen_pair, 0)
laplace_prob = (cnt_w1w2 + 1) / (unigram_counts[w1] + V)
print(f"Example unseen bigram: {unseen_pair}")
print(f"Count({w1},{w2}) = {cnt_w1w2}, Count({w1}) = {unigram_counts[w1]}")
print(f"P_Laplace({w2}|{w1}) = (0 + 1) / ({unigram_counts[w1]} + {V}) = 1/{unigram_counts[w1] + V} = {laplace_prob:.4f}")
print()

# ------------------------------------------------------------
# 3. Count-of-counts table
# N_c = number of distinct bigrams that occur exactly c times
# N_0 = V^2 - number of distinct observed bigrams
# ------------------------------------------------------------
print("=" * 70)
print("3. COUNT-OF-COUNTS TABLE")
print("=" * 70)

observed_counts = list(bigram_counts.values())
count_freq = Counter(observed_counts)   # c -> how many bigrams have that count
distinct_bigrams = len(bigram_counts)
N0 = V * V - distinct_bigrams

print(f"{'c':<6} {'N_c':<6}")
print("-" * 20)
print(f"{0:<6} {N0:<6}")
for c in sorted(count_freq):
    print(f"{c:<6} {count_freq[c]:<6}")
print()

# ------------------------------------------------------------
# 4. Good-Turing discounting
# Adjusted count c* = (c+1) * N_{c+1} / N_c
# If any N_{c+1} needed is zero, smooth the N_c table by adding 1.
# ------------------------------------------------------------
print("=" * 70)
print("4. GOOD-TURING DISCOUNTING")
print("=" * 70)

max_c = max(observed_counts) if observed_counts else 0
# Build original N_c dictionary for c=0..max_c+1 (we need N_{max_c+1})
Nc_orig = {}
for c in range(0, max_c + 2):
    if c == 0:
        Nc_orig[c] = N0
    else:
        Nc_orig[c] = count_freq.get(c, 0)

# Determine if smoothing is needed: for any observed bigram count c,
# we need N_{c+1} and if it is 0 (or missing) then smooth.
need_smoothing = False
for c in set(observed_counts):
    if Nc_orig.get(c + 1, 0) == 0:
        need_smoothing = True
        break

if need_smoothing:
    print("Note: N_{c+1} missing or zero for some c. Applying add-one smoothing to the count-of-counts table.\n")
    # Add-one smoothing: N'_c = N_c + 1 for all c from 0 to max_c+1
    Nc_smooth = {c: Nc_orig[c] + 1 for c in range(0, max_c + 2)}
else:
    Nc_smooth = Nc_orig.copy()

# Compute c* for all c in [0, max_c]
c_star = {}
for c in range(0, max_c + 1):
    if Nc_smooth[c] == 0:
        c_star[c] = 0.0
    else:
        c_star[c] = (c + 1) * Nc_smooth[c + 1] / Nc_smooth[c]

# Print the Good-Turing adjusted bigram table
print(f"{'w1':<12} {'w2':<12} {'orig c':<8} {'c*':<10} {'P_GT':<10}")
print("-" * 60)
for (w1, w2), cnt in sorted(bigram_counts.items()):
    c_orig = cnt
    c_adj = c_star[c_orig]
    prob_gt = c_adj / total_bigram_tokens
    print(f"{w1:<12} {w2:<12} {c_orig:<8} {c_adj:<10.4f} {prob_gt:<10.6f}")
print()

# ------------------------------------------------------------
# 5. Probability of a missing bigram (c=0) using Good-Turing
# P(unseen) = c*(0) / N   where c*(0) = (0+1)*N_1 / N_0
# ------------------------------------------------------------
print("=" * 70)
print("5. PROBABILITY OF A MISSING BIGRAM (GOOD-TURING)")
print("=" * 70)
c0_star = (0 + 1) * Nc_smooth[1] / Nc_smooth[0]
prob_missing = c0_star / total_bigram_tokens
print(f"N'_0 = {Nc_smooth[0]}, N'_1 = {Nc_smooth[1]}")
print(f"c*(0) = 1 * N'_1 / N'_0 = {Nc_smooth[1]} / {Nc_smooth[0]} = {c0_star:.6f}")
print(f"P(unseen bigram) = c*(0) / N = {c0_star:.6f} / {total_bigram_tokens} = {prob_missing:.6f}")
print()

# ------------------------------------------------------------
# 6. Explanation
# ------------------------------------------------------------
print("=" * 70)
print("6. EXPLANATION")
print("=" * 70)
explanation = (
    "Smoothing the count-of-counts table (adding 1 to each Nc) prevents "
    "the Good‑Turing formula from breaking when we need N_{c+1} for a "
    "count c but that frequency bin is empty or zero. For example, the "
    "highest observed count c_max always requires N_{c_max+1} which is "
    "zero, leading to a zero adjusted count. By adding a small constant "
    "to every Nc we ensure all required N_{c+1} are positive, giving "
    "non‑zero probability estimates for rare and unseen events. This "
    "stabilizes the reallocation of probability mass and yields a more "
    "reliable language model."
)
print(explanation)
