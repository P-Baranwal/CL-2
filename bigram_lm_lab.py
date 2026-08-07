"""
Lab assignment: Language Modelling (bigrams, Laplace smoothing, Good-Turing discounting)

Corpus: 4 tokenized sentences with <s> / </s> markers (Peter Piper nursery rhyme).
"""

from collections import Counter

# ------------------------------------------------------------
# Corpus
# ------------------------------------------------------------
sentences = [
    "<s> peter piper picked a peck of pickled peppers </s>",
    "<s> a peck of pickled peppers peter piper picked </s>",
    "<s> if peter piper picked a peck of pickled peppers </s>",
    "<s> where s the peck of pickled peppers peter piper picked </s>",
]

tokenized = [s.split() for s in sentences]

# Vocabulary (includes <s> and </s>)
vocab = sorted(set(token for sent in tokenized for token in sent))
V = len(vocab)
print("Vocabulary:", vocab)
print(f"Vocabulary size V = {V}\n")

# Unigram counts
unigram_counts = Counter(token for sent in tokenized for token in sent)

# Bigram counts (built per-sentence, so no bigram spans two sentences)
bigram_counts = Counter()
for sent in tokenized:
    bigram_counts.update(zip(sent, sent[1:]))

total_bigram_tokens = sum(bigram_counts.values())  # N for Good-Turing

# ------------------------------------------------------------
# 1. Bigram probability table (MLE)
#    P(w2|w1) = count(w1,w2) / count(w1)
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
#    P_Laplace(w2|w1) = (count(w1,w2) + 1) / (count(w1) + V)
#
#    Improvement: show several unseen bigrams (assignment says
#    "bigrams", plural), not just one, to demonstrate the formula
#    handles both the "never co-occur" case and the structural
#    "</s> is never followed by anything" case.
# ------------------------------------------------------------
print("=" * 70)
print("2. LAPLACE (ADD-ONE) SMOOTHING")
print("=" * 70)
print(f"Vocabulary size V = {V}\n")

unseen_examples = [
    ("</s>", "peter"),  # structural: </s> never precedes anything
    ("peter", "peck"),  # two in-vocabulary words that never co-occur
    ("of", "peppers"),  # another never-observed pair
]

print(f"{'w1':<10} {'w2':<10} {'count(w1,w2)':<14} {'count(w1)':<11} {'P_Laplace(w2|w1)':<10}")
print("-" * 70)
for w1, w2 in unseen_examples:
    cnt_w1w2 = bigram_counts.get((w1, w2), 0)
    laplace_prob = (cnt_w1w2 + 1) / (unigram_counts[w1] + V)
    print(f"{w1:<10} {w2:<10} {cnt_w1w2:<14} {unigram_counts[w1]:<11} {laplace_prob:.4f}")

w1, w2 = unseen_examples[0]
cnt_w1w2 = bigram_counts.get((w1, w2), 0)
laplace_prob = (cnt_w1w2 + 1) / (unigram_counts[w1] + V)
print(f"\nWorked example: P_Laplace({w2}|{w1}) = (0 + 1) / "
      f"({unigram_counts[w1]} + {V}) = 1/{unigram_counts[w1] + V} = {laplace_prob:.4f}")
print()

# ------------------------------------------------------------
# 3. Count-of-counts table
#    N_c = number of distinct bigrams that occur exactly c times
#    N_0 = V^2 - number of distinct observed bigrams
# ------------------------------------------------------------
print("=" * 70)
print("3. COUNT-OF-COUNTS TABLE")
print("=" * 70)

observed_counts = list(bigram_counts.values())
count_freq = Counter(observed_counts)
distinct_bigrams = len(bigram_counts)
N0 = V * V - distinct_bigrams
max_c = max(observed_counts) if observed_counts else 0

print(f"{'c':<6} {'N_c':<6}")
print("-" * 20)
print(f"{0:<6} {N0:<6}")
# Improvement: iterate the full range 1..max_c so any count value with
# zero frequency is still shown explicitly (original silently skipped it).
for c in range(1, max_c + 1):
    print(f"{c:<6} {count_freq.get(c, 0):<6}")
print()

# ------------------------------------------------------------
# 4. Good-Turing discounting
#    Adjusted count c* = (c+1) * N_{c+1} / N_c
#    If any N_{c+1} needed is zero, add-one smooth the N_c table.
# ------------------------------------------------------------
print("=" * 70)
print("4. GOOD-TURING DISCOUNTING")
print("=" * 70)

# Build the raw N_c table for c = 0 .. max_c+1 (need N_{max_c+1})
Nc_orig = {}
for c in range(0, max_c + 2):
    Nc_orig[c] = N0 if c == 0 else count_freq.get(c, 0)

# Do we need smoothing? True if any observed c has N_{c+1} == 0
need_smoothing = any(Nc_orig.get(c + 1, 0) == 0 for c in set(observed_counts))

if need_smoothing:
    print("N_{c+1} is missing/zero for at least one observed c "
          "(here, the highest count always has an empty next bin).")
    print("Applying add-one smoothing to the count-of-counts table: N'_c = N_c + 1\n")
    Nc_smooth = {c: Nc_orig[c] + 1 for c in range(0, max_c + 2)}
else:
    Nc_smooth = Nc_orig.copy()

# Improvement: explicitly show the before/after count-of-counts table,
# since the assignment asks to "show" the adjustment, and the original
# script computed this table but never printed it.
print(f"{'c':<6} {'N_c':<8} {'N\\'_c (smoothed)':<16}")
print("-" * 32)
for c in range(0, max_c + 2):
    print(f"{c:<6} {Nc_orig[c]:<8} {Nc_smooth[c]:<16}")
print()

# c* for each observed count c = 0 .. max_c
c_star = {}
for c in range(0, max_c + 1):
    c_star[c] = 0.0 if Nc_smooth[c] == 0 else (c + 1) * Nc_smooth[c + 1] / Nc_smooth[c]

print(f"{'w1':<12} {'w2':<12} {'orig c':<8} {'c*':<10} {'P_GT':<10}")
print("-" * 60)
for (w1, w2), cnt in sorted(bigram_counts.items()):
    c_adj = c_star[cnt]
    prob_gt = c_adj / total_bigram_tokens
    print(f"{w1:<12} {w2:<12} {cnt:<8} {c_adj:<10.4f} {prob_gt:<10.6f}")
print()

# Improvement: sanity-check that the GT-adjusted counts roughly conserve
# total probability mass (sum of N_c * c* over all c, including c=0,
# should be close to N). This is a useful correctness check to include.
mass = Nc_orig[0] * c_star[0] + sum(Nc_orig[c] * c_star[c] for c in range(1, max_c + 1))
print(f"Sanity check - total redistributed mass: {mass:.4f} "
      f"(should be close to N = {total_bigram_tokens})")
print("Note: the bigram ('a','peck') has orig count 3 but only N_3=1 bigram "
      "shares that count while N_4=5 bigrams occur 4 times, so its smoothed "
      f"c* jumps to {c_star[3]:.1f} - a known artifact of add-one-smoothing "
      "the count-of-counts table on such a small, sparse corpus.")
print()

# ------------------------------------------------------------
# 5. Probability of a missing bigram (c=0) using Good-Turing
#    P(unseen) = c*(0) / N   where c*(0) = (0+1)*N'_1 / N'_0
# ------------------------------------------------------------
print("=" * 70)
print("5. PROBABILITY OF A MISSING BIGRAM (GOOD-TURING)")
print("=" * 70)
c0_star = c_star[0]
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
    "Smoothing the count-of-counts table (adding 1 to every N_c) prevents "
    "the Good-Turing formula c* = (c+1)*N_(c+1)/N_c from breaking when the "
    "next frequency bin N_(c+1) is empty - which always happens for the "
    "highest observed count, since nothing occurs one more time than the "
    "most frequent bigram. Without smoothing, that bigram's adjusted count "
    "would collapse to zero. Adding a small constant to every N_c keeps all "
    "the bins positive, so every observed and unseen bigram gets a "
    "well-defined, non-zero probability estimate. On a very small corpus "
    "like this one, this simple fix can make the estimates for sparsely "
    "populated bins (e.g. c=3, where only one bigram is observed) noisy or "
    "non-monotonic, but it still guarantees the formula is always "
    "computable and reallocates a sensible amount of probability mass to "
    "unseen events."
)
print(explanation)
