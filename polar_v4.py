"""
Polar DNA Compressor — V4

Two lossless codecs, both using an order-k Markov model + range (arithmetic) coder:
  • V4-plain:  predicts base x_i from the previous k bases alone.
  • V4-polar:  same, but ALSO conditions on the quantized complex-centroid
               of the current block (a "polar context").

V4-polar is the honest test of whether the polar transform adds predictive
information BEYOND what a pure k-gram context model already sees.

Run:  python3 polar_v4.py
"""

import numpy as np
import random, time, sys, os, zlib, bz2, lzma
import constriction
RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

# ---------------- data generators ----------------

def random_dna(n, seed=0):
    return "".join(random.Random(seed).choices(BASES, k=n))

def skewed_dna(n, seed=1):
    return "".join(random.Random(seed).choices(BASES, weights=(0.4,0.1,0.1,0.4), k=n))

def repeat_dna(n, motif="GATTACA"):
    return (motif * (n // len(motif) + 1))[:n]

# Short real human mtDNA segment (~200 bp sample the user posted earlier)
MTDNA_SAMPLE = (
    "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTTCGTCTGGGGGGTATGCACGCGATAGCATTGCGAGACGCTGG"
    "AGCCGGAGCACCCTATGTCGCAGTATCTGTCTTTGATTCCTGCCTCATCCTATTATTTATCGCACCTACGTTCAATATTACAGGCGAACATACTTACTAAA"
) * 250  # tile to ~50 kb so compression has room to work

# ---------------- helpers ----------------

def seq_to_ids(seq):
    return np.array([B2I[b] for b in seq], dtype=np.int32)

def ids_to_seq(ids):
    return "".join(BASES[i] for i in ids)

def pack_2bit(seq):
    ids = seq_to_ids(seq)
    pad = (-len(ids)) % 4
    if pad:
        ids = np.concatenate([ids, np.zeros(pad, dtype=np.int32)])
    b = (ids[0::4] << 6) | (ids[1::4] << 4) | (ids[2::4] << 2) | ids[3::4]
    return b.astype(np.uint8).tobytes()

# ---------------- Markov model ----------------

def build_markov(ids, k, n_contexts_by_centroid=None, centroid_ctx=None, smoothing=1.0):
    """
    Build an order-k Markov model giving P(x_i | last k bases [, centroid_ctx]).
    If centroid_ctx is provided (same length as ids), the model becomes
        P(x_i | context_k, centroid_ctx[i])
    Returns a dict mapping (context_tuple, [c_bucket]) -> np.array length 4 of probs.
    """
    if centroid_ctx is None:
        counts = {}
        for i in range(k, len(ids)):
            ctx = tuple(int(x) for x in ids[i-k:i])
            if ctx not in counts:
                counts[ctx] = np.full(4, smoothing)
            counts[ctx][ids[i]] += 1
        return {c: v / v.sum() for c, v in counts.items()}
    else:
        counts = {}
        for i in range(k, len(ids)):
            ctx = (tuple(int(x) for x in ids[i-k:i]), int(centroid_ctx[i]))
            if ctx not in counts:
                counts[ctx] = np.full(4, smoothing)
            counts[ctx][ids[i]] += 1
        return {c: v / v.sum() for c, v in counts.items()}

# ---------------- centroid context (polar side) ----------------

def block_centroid_context(seq, block_size=16, n_buckets=16):
    """For each position i, return the quantized angle-bucket of its containing block's centroid."""
    z = np.exp(1j * PHASES[seq_to_ids(seq)])
    pad = (-len(z)) % block_size
    if pad:
        z = np.concatenate([z, np.zeros(pad, dtype=complex)])
    blocks = z.reshape(-1, block_size)
    centroids = np.angle(blocks.sum(axis=1)) % (2 * np.pi)
    buckets = np.floor(centroids * n_buckets / (2 * np.pi)).astype(np.int32)
    buckets = np.clip(buckets, 0, n_buckets - 1)
    # broadcast: each base inside block j gets the same bucket
    per_base = np.repeat(buckets, block_size)[:len(seq)]
    return per_base, buckets  # buckets itself is overhead stored as side info

# ---------------- range coding with variable model per symbol ----------------

def encode_with_model(ids, k, model, uniform_prior=None, centroid_ctx=None):
    """Encode ids using given model dict. First k bases are sent with a uniform prior."""
    enc = RangeEncoder()
    # First k bases: uniform over 4 symbols
    uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    for i in range(min(k, len(ids))):
        enc.encode(np.asarray([ids[i]], dtype=np.int32),
                   Categorical(uniform, perfect=False))
    # Rest: conditional probs
    for i in range(k, len(ids)):
        if centroid_ctx is not None:
            ctx = (tuple(int(x) for x in ids[i-k:i]), int(centroid_ctx[i]))
        else:
            ctx = tuple(int(x) for x in ids[i-k:i])
        p = model.get(ctx, uniform).astype(np.float64)
        p = p / p.sum()  # renormalize after any floating slop
        enc.encode(np.asarray([ids[i]], dtype=np.int32),
                   Categorical(p, perfect=False))
    compressed = enc.get_compressed().tobytes()
    return compressed

def decode_with_model(compressed, n, k, model, centroid_ctx=None):
    """Decode a compressed bitstream back to an id array."""
    arr = np.frombuffer(compressed, dtype=np.uint32).copy()
    dec = RangeDecoder(arr)
    uniform = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float64)
    out = np.zeros(n, dtype=np.int32)
    for i in range(min(k, n)):
        sym = dec.decode(Categorical(uniform, perfect=False))
        out[i] = int(sym[0]) if hasattr(sym, '__len__') else int(sym)
    for i in range(k, n):
        if centroid_ctx is not None:
            ctx = (tuple(int(x) for x in out[i-k:i]), int(centroid_ctx[i]))
        else:
            ctx = tuple(int(x) for x in out[i-k:i])
        p = model.get(ctx, uniform).astype(np.float64)
        p = p / p.sum()
        sym = dec.decode(Categorical(p, perfect=False))
        out[i] = int(sym[0]) if hasattr(sym, '__len__') else int(sym)
    return out

# ---------------- end-to-end codecs ----------------

def compress_v4_plain(seq, k=4):
    ids = seq_to_ids(seq)
    model = build_markov(ids, k)
    # For fairness, we must ALSO send the model so the decoder can reconstruct.
    # We'll estimate its overhead: 4**k contexts * 4 counts * 2 bytes each.
    model_overhead = (4 ** k) * 4 * 2  # bytes
    payload = encode_with_model(ids, k, model)
    # Side info: seq length (4 bytes)
    return 4 + model_overhead + len(payload), len(payload), model_overhead

def compress_v4_polar(seq, k=4, block_size=16, n_buckets=16):
    ids = seq_to_ids(seq)
    cctx, bucket_stream = block_centroid_context(seq, block_size, n_buckets)
    model = build_markov(ids, k, centroid_ctx=cctx)
    payload = encode_with_model(ids, k, model, centroid_ctx=cctx)
    # Side info: seq length, block size, n_buckets, + bucket stream (packed at log2(n_buckets) bits each)
    bits_per_bucket = int(np.ceil(np.log2(n_buckets)))
    bucket_side = int(np.ceil(len(bucket_stream) * bits_per_bucket / 8))
    # Model overhead: (4**k) * n_buckets * 4 * 2 bytes
    model_overhead = (4 ** k) * n_buckets * 4 * 2
    return 8 + model_overhead + bucket_side + len(payload), len(payload), model_overhead + bucket_side

# ---------------- benchmark ----------------

def bench(seq, label, k=4):
    n = len(seq)
    baseline = (n + 3) // 4
    bz = len(bz2.compress(seq.encode(), 9))
    lz = len(lzma.compress(seq.encode(), preset=9))

    total_plain, pl_payload, pl_over = compress_v4_plain(seq, k=k)
    total_polar, po_payload, po_over = compress_v4_polar(seq, k=k)

    def bpb(sz): return 8 * sz / n
    def ratio(sz): return baseline / sz

    print(f"\n{'='*78}")
    print(f"{label}  (n = {n:,} bases,  k = {k})")
    print(f"{'='*78}")
    print(f"{'Method':<40} {'Bytes':>10} {'Bits/base':>12} {'vs 2-bit':>10}")
    print("-" * 78)
    print(f"{'2-bit packed baseline':<40} {baseline:>10} {bpb(baseline):>12.3f} {ratio(baseline):>9.2f}x")
    print(f"{'Raw ASCII + bz2':<40} {bz:>10} {bpb(bz):>12.3f} {ratio(bz):>9.2f}x")
    print(f"{'Raw ASCII + lzma':<40} {lz:>10} {bpb(lz):>12.3f} {ratio(lz):>9.2f}x")
    print(f"{'V4-plain  payload only':<40} {pl_payload:>10} {bpb(pl_payload):>12.3f} {ratio(pl_payload):>9.2f}x")
    print(f"{'V4-plain  + model overhead':<40} {total_plain:>10} {bpb(total_plain):>12.3f} {ratio(total_plain):>9.2f}x")
    print(f"{'V4-polar  payload only':<40} {po_payload:>10} {bpb(po_payload):>12.3f} {ratio(po_payload):>9.2f}x")
    print(f"{'V4-polar  + model + bucket overhead':<40} {total_polar:>10} {bpb(total_polar):>12.3f} {ratio(total_polar):>9.2f}x")
    return pl_payload, po_payload

def roundtrip_check(seq, k=4):
    """Verify V4-plain decodes exactly back to the original."""
    ids = seq_to_ids(seq)
    model = build_markov(ids, k)
    payload = encode_with_model(ids, k, model)
    decoded = decode_with_model(payload, len(ids), k, model)
    ok = np.array_equal(ids, decoded)
    return ok

if __name__ == "__main__":
    # Round-trip sanity check on a small sequence first
    small = random_dna(2000, seed=42)
    print(f"Round-trip check (V4-plain, 2k random bases): {'OK' if roundtrip_check(small) else 'FAIL'}")

    N = 100_000
    bench(random_dna(N),        "RANDOM DNA",        k=4)
    bench(skewed_dna(N),        "AT-RICH DNA",       k=4)
    bench(repeat_dna(N),        "REPETITIVE DNA",    k=4)
    bench(MTDNA_SAMPLE,         "HUMAN mtDNA (tiled)", k=4)
