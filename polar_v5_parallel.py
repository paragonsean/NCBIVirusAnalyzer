"""
polar_v5_parallel.py  --  Block-parallel adaptive DNA compressor.

Key idea: split the sequence into independent tiles, encode each with its
own adaptive PPM model.  Each tile is embarrassingly parallel (a GPU or
thread pool encodes all tiles concurrently).  The cost is warm-up waste:
each tile's model starts cold, so the first ~4^k symbols in each tile
compress poorly.

We measure the compression-ratio cost of tiling at various tile sizes
(64, 256, 1024, 4096, 16384) vs. monolithic V5, to quantify how much
ratio you trade for parallelism — the number you'd need before investing
in a Mojo/GPU rewrite.
"""

import numpy as np, random, zlib, bz2, lzma, struct, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import constriction

RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical  = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
I2B = {i: b for i, b in enumerate(BASES)}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

# --------------- utilities ---------------
def seq_to_ids(s):
    return np.fromiter((B2I[c] for c in s), dtype=np.int32, count=len(s))

def ids_to_seq(ids):
    return "".join(I2B[int(i)] for i in ids)

# --------------- sources ---------------
def random_dna(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice(BASES) for _ in range(n))

def skewed_dna(n, seed=1):
    rng = random.Random(seed); w = [0.35,0.15,0.15,0.35]
    return "".join(rng.choices(BASES, w)[0] for _ in range(n))

def repeat_dna(n, motif="GATTACA"):
    return (motif * (n // len(motif) + 1))[:n]

MTDNA = ("GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTT"
         "CGTCTGGGGGGTGTGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTC"
         "GCAGTATCTGTCTTTGATTCCTGCCTCATTCTATTATTTATCGCACCTACGTTCAATATT"
         "ACAGGCGAACATACCTACTAAAGTGTGTTAATTAATTAATGCTTGTAGGACATAATAATA"
         "ACAATTGAATGTCTGCACAGCCGCTTTCCACACAGACATCATAACAAAAAATTTCCACCA"
         "AACCCCCCCCTCCCCCGCTTCTGGCCACAGCACTTAAACACATCTCTGCCAAACCCCAAA"
         "AACAAAGAACCCTAACACCAGCCTAACCAGATTTCAAACTTTATCTTTTGGCGGTATGCA")
MTDNA_TILE = (MTDNA * 150)[:50000]

# ================================================================
# ADAPTIVE PPM MODEL (same as V5, self-contained)
# ================================================================
def make_probs(counts_stack, alpha=1.0):
    p = np.full(4, 0.25, dtype=np.float64)
    for c in counts_stack:
        total = c.sum()
        p = (c + alpha * p) / (total + alpha)
    p = np.clip(p, 1e-9, None); p /= p.sum()
    return p

class AdaptiveModel:
    def __init__(self, k, alpha=1.0):
        self.k = k; self.alpha = alpha
        self.tables = [dict() for _ in range(k + 1)]
    def probs(self, ctx_full):
        stack = []
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])
            stack.append(self.tables[o].get(ctx, np.zeros(4, dtype=np.int64)))
        return make_probs(stack, self.alpha)
    def update(self, ctx_full, sym):
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])
            arr = self.tables[o].get(ctx)
            if arr is None:
                arr = np.zeros(4, dtype=np.int64)
                self.tables[o][ctx] = arr
            arr[sym] += 1

# ================================================================
# SINGLE-TILE ENCODE / DECODE  (the unit of parallelism)
# ================================================================
def encode_tile(ids, k=4, alpha=1.0):
    """Encode a single tile of ids -> bytes. Returns compressed bytes."""
    enc = RangeEncoder()
    model = AdaptiveModel(k, alpha)
    pad = np.zeros(k, dtype=np.int32)
    for i in range(len(ids)):
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        p = model.probs(ctx)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]))
    return enc.get_compressed().tobytes()

def decode_tile(payload_bytes, n, k=4, alpha=1.0):
    """Decode a single tile from bytes -> ids array."""
    arr = np.frombuffer(payload_bytes, dtype=np.uint8).copy()
    pad_bytes = (-len(arr)) % 4
    if pad_bytes:
        arr = np.concatenate([arr, np.zeros(pad_bytes, dtype=np.uint8)])
    u32 = arr.view(np.uint32).copy()
    dec = RangeDecoder(u32)
    model = AdaptiveModel(k, alpha)
    pad = np.zeros(k, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ctx = np.concatenate([pad, out[:i]])[-k:]
        p = model.probs(ctx)
        sym = dec.decode(Categorical(p, perfect=False))
        out[i] = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
        model.update(ctx, int(out[i]))
    return out

# ================================================================
# POLAR-CONDITIONED TILE (V5-polar per-tile)
# ================================================================
class AdaptivePolarModel:
    def __init__(self, k, n_buckets, alpha=1.0):
        self.k = k; self.n_buckets = n_buckets; self.alpha = alpha
        self.banks = [AdaptiveModel(k, alpha) for _ in range(n_buckets)]
    def probs(self, ctx, b): return self.banks[b].probs(ctx)
    def update(self, ctx, sym, b): self.banks[b].update(ctx, sym)

def encode_tile_polar(ids, k=4, block_size=64, n_buckets=8, alpha=1.0):
    enc = RangeEncoder()
    model = AdaptivePolarModel(k, n_buckets, alpha)
    z_vals = np.exp(1j * PHASES[ids])
    pad = np.zeros(k, dtype=np.int32)
    window = []
    for i in range(len(ids)):
        if not window: b = 0
        else:
            c = sum(window) / len(window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        p = model.probs(ctx, b)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]), b)
        window.append(z_vals[i])
        if len(window) > block_size: window.pop(0)
    return enc.get_compressed().tobytes()

def decode_tile_polar(payload_bytes, n, k=4, block_size=64, n_buckets=8, alpha=1.0):
    arr = np.frombuffer(payload_bytes, dtype=np.uint8).copy()
    pad_bytes = (-len(arr)) % 4
    if pad_bytes:
        arr = np.concatenate([arr, np.zeros(pad_bytes, dtype=np.uint8)])
    u32 = arr.view(np.uint32).copy()
    dec = RangeDecoder(u32)
    model = AdaptivePolarModel(k, n_buckets, alpha)
    pad = np.zeros(k, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    window = []
    for i in range(n):
        if not window: b = 0
        else:
            c = sum(window) / len(window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        ctx = np.concatenate([pad, out[:i]])[-k:]
        p = model.probs(ctx, b)
        sym = dec.decode(Categorical(p, perfect=False))
        out[i] = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
        model.update(ctx, int(out[i]), b)
        window.append(np.exp(1j * PHASES[out[i]]))
        if len(window) > block_size: window.pop(0)
    return out

# ================================================================
# BLOCK-PARALLEL CODEC
# ================================================================
def compress_parallel(seq, tile_size, k=4, polar=False, n_buckets=8,
                      max_workers=4):
    """Split seq into tiles, encode each independently, return packed bytes."""
    ids = seq_to_ids(seq)
    n = len(ids)
    n_tiles = (n + tile_size - 1) // tile_size
    tiles = [ids[i*tile_size : (i+1)*tile_size] for i in range(n_tiles)]

    # Header: 4 bytes n_tiles, then per-tile: 4 bytes tile_len, 4 bytes payload_len
    enc_fn = (lambda t: encode_tile_polar(t, k, 64, n_buckets)) if polar \
             else (lambda t: encode_tile(t, k))

    # Encode tiles (use thread pool to simulate GPU parallelism)
    payloads = [None] * n_tiles
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(enc_fn, t): i for i, t in enumerate(tiles)}
        for f in as_completed(futs):
            payloads[futs[f]] = f.result()

    # Pack: header + payloads
    parts = [struct.pack("<I", n_tiles)]
    for i in range(n_tiles):
        parts.append(struct.pack("<II", len(tiles[i]), len(payloads[i])))
    for p in payloads:
        parts.append(p)
    return b"".join(parts)

def decompress_parallel(data, k=4, polar=False, n_buckets=8, max_workers=4):
    """Unpack and decode all tiles, return full sequence string."""
    off = 0
    n_tiles = struct.unpack_from("<I", data, off)[0]; off += 4
    tile_meta = []
    for _ in range(n_tiles):
        tlen, plen = struct.unpack_from("<II", data, off); off += 8
        tile_meta.append((tlen, plen))

    dec_fn = (lambda pay, tl: decode_tile_polar(pay, tl, k, 64, n_buckets)) \
             if polar else (lambda pay, tl: decode_tile(pay, tl, k))

    payloads = []
    for tlen, plen in tile_meta:
        payloads.append(data[off:off+plen]); off += plen

    results = [None] * n_tiles
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(dec_fn, payloads[i], tile_meta[i][0]): i
                for i in range(n_tiles)}
        for f in as_completed(futs):
            results[futs[f]] = f.result()

    return ids_to_seq(np.concatenate(results))

# ================================================================
# MONOLITHIC V5 (for comparison, no tiling)
# ================================================================
def compress_mono(seq, k=4, polar=False, n_buckets=8):
    ids = seq_to_ids(seq)
    if polar:
        return encode_tile_polar(ids, k, 64, n_buckets)
    return encode_tile(ids, k)

# ================================================================
# BENCHMARK
# ================================================================
def bench(seq, label, k=4, tile_sizes=(64, 256, 1024, 4096, 16384)):
    ids = seq_to_ids(seq)
    n = len(ids)
    raw = seq.encode()
    base_2bit = (n + 3) // 4

    print(f"\n{'='*90}")
    print(f"{label}  (n = {n:,} bases,  k = {k})")
    print(f"{'='*90}")
    print(f"{'Method':<52}{'Bytes':>10}{'Bits/base':>10}{'vs 2bit':>9}{'Time':>9}")
    print("-" * 90)

    def row(name, b, t=0):
        ts = f"{t:.2f}s" if t else ""
        print(f"{name:<52}{b:>10}{b*8/n:>10.3f}{base_2bit/max(b,1):>8.1f}x{ts:>9}")

    row("2-bit packed baseline", base_2bit)
    row("Raw ASCII + bz2", len(bz2.compress(raw)))
    row("Raw ASCII + lzma", len(lzma.compress(raw)))

    # Monolithic V5-plain
    t0 = time.time()
    mono = compress_mono(seq, k=k, polar=False)
    t1 = time.time()
    row(f"V5 monolithic (k={k})", len(mono), t1-t0)

    # Monolithic V5-polar
    t0 = time.time()
    mono_p = compress_mono(seq, k=k, polar=True, n_buckets=16)
    t1 = time.time()
    row(f"V5 monolithic polar (k={k}, B=16)", len(mono_p), t1-t0)

    # Block-parallel at various tile sizes
    for ts in tile_sizes:
        if ts > n: continue
        n_tiles = (n + ts - 1) // ts
        t0 = time.time()
        blob = compress_parallel(seq, ts, k=k, polar=False, max_workers=4)
        t1 = time.time()
        row(f"V5-parallel tiles={ts:>5} ({n_tiles:>3} tiles)", len(blob), t1-t0)

    # Block-parallel polar at best tile size
    for ts in [1024, 4096]:
        if ts > n: continue
        n_tiles = (n + ts - 1) // ts
        t0 = time.time()
        blob = compress_parallel(seq, ts, k=k, polar=True, n_buckets=16,
                                 max_workers=4)
        t1 = time.time()
        row(f"V5-parallel polar tiles={ts:>5} ({n_tiles:>3} tiles, B=16)",
            len(blob), t1-t0)

def roundtrip(label, seq, tile_size=256, k=4):
    blob = compress_parallel(seq, tile_size, k=k, polar=False)
    dec = decompress_parallel(blob, k=k, polar=False)
    ok1 = (dec == seq)
    blob2 = compress_parallel(seq, tile_size, k=k, polar=True, n_buckets=8)
    dec2 = decompress_parallel(blob2, k=k, polar=True, n_buckets=8)
    ok2 = (dec2 == seq)
    print(f"Round-trip {label} (tile={tile_size}): plain={'OK' if ok1 else 'FAIL'}"
          f"  polar={'OK' if ok2 else 'FAIL'}")

if __name__ == "__main__":
    # Round-trip checks
    roundtrip("random 1k", random_dna(1000, seed=7), tile_size=256)
    roundtrip("AT-rich 1k", skewed_dna(1000, seed=8), tile_size=256)
    roundtrip("mtDNA 1k", MTDNA_TILE[:1000], tile_size=256)

    N = 20_000   # smaller N since we're running many configs
    bench(random_dna(N),        "RANDOM DNA",        k=4)
    bench(skewed_dna(N),        "AT-RICH DNA",       k=4)
    bench(repeat_dna(N),        "REPETITIVE DNA",    k=4)
    bench(MTDNA_TILE[:N],       "HUMAN mtDNA (tiled)", k=4)
