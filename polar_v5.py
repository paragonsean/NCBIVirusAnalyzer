"""
polar_v5.py  --  Adaptive arithmetic coding for DNA.

Key improvements over V4:
  * ONLINE adaptive model -- no static table is transmitted (zero model overhead)
  * PPM-style backoff: order-k -> order-(k-1) -> ... -> order-0 -> uniform
  * Optional polar-centroid conditioning (V5-polar) with adaptive conditional tables
  * Optional reverse-complement canonicalization (V5-rc) on biological data

V5-plain is apples-to-apples vs V4-plain's "+ model overhead" line.
"""

import numpy as np, zlib, bz2, lzma, random, argparse, json, struct
from pathlib import Path
import constriction

RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical  = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
I2B = {i: b for i, b in enumerate(BASES)}
RC  = {"A":"T","T":"A","C":"G","G":"C"}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
MAGIC = b"PQC1"

# --------------- utilities ---------------
def seq_to_ids(s):
    return np.fromiter((B2I[c] for c in s), dtype=np.int32, count=len(s))

def ids_to_seq(ids):
    return "".join(I2B[int(i)] for i in ids)

def reverse_complement(s):
    return "".join(RC[c] for c in reversed(s))

def canonical(s):
    rc = reverse_complement(s)
    return (s, False) if s <= rc else (rc, True)

# --------------- sources ---------------
def random_dna(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choice(BASES) for _ in range(n))

def skewed_dna(n, seed=1):
    rng = random.Random(seed); w = [0.35,0.15,0.15,0.35]
    return "".join(rng.choices(BASES, w)[0] for _ in range(n))

def repeat_dna(n, motif="GATTACA"):
    return (motif * (n // len(motif) + 1))[:n]

# Human mtDNA fragment (real biology)
MTDNA = ("GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGCATTTGGTATTTT"
         "CGTCTGGGGGGTGTGCACGCGATAGCATTGCGAGACGCTGGAGCCGGAGCACCCTATGTC"
         "GCAGTATCTGTCTTTGATTCCTGCCTCATTCTATTATTTATCGCACCTACGTTCAATATT"
         "ACAGGCGAACATACCTACTAAAGTGTGTTAATTAATTAATGCTTGTAGGACATAATAATA"
         "ACAATTGAATGTCTGCACAGCCGCTTTCCACACAGACATCATAACAAAAAATTTCCACCA"
         "AACCCCCCCCTCCCCCGCTTCTGGCCACAGCACTTAAACACATCTCTGCCAAACCCCAAA"
         "AACAAAGAACCCTAACACCAGCCTAACCAGATTTCAAACTTTATCTTTTGGCGGTATGCA")
MTDNA_TILE = (MTDNA * 150)[:50000]

# =================================================================
# ADAPTIVE ORDER-k WITH PPM-STYLE BACKOFF
# =================================================================
#
# At each position i, encoder and decoder share the same state:
#   * counts[order][ctx] -> np.array of 4 counts
# To predict symbol i:
#   1. Look up ctx = seq[i-k : i] in order-k. If all 4 counts > 0, use it.
#   2. Otherwise mix with order-(k-1), ..., order-0, uniform, using escape.
# For simplicity and correctness we use **interpolated smoothing**:
#   p(s | ctx_k) = (counts_k[s] + alpha * p(s | ctx_{k-1})) / (sum_k + alpha)
# with p at order -1 = uniform 1/4.

def make_probs(counts_stack, alpha=1.0):
    """counts_stack: list from order-0 up to order-k. Each is a length-4 array."""
    p = np.full(4, 0.25, dtype=np.float64)   # order -1 (uniform)
    for c in counts_stack:
        total = c.sum()
        p = (c + alpha * p) / (total + alpha)
    # Guarantee strictly positive, normalize
    p = np.clip(p, 1e-9, None); p = p / p.sum()
    return p

class AdaptiveModel:
    """Order-k with interpolated smoothing, updated online."""
    def __init__(self, k, alpha=1.0):
        self.k = k
        self.alpha = alpha
        # tables[o] : dict ctx(tuple len o) -> np.array(4)
        self.tables = [dict() for _ in range(k + 1)]

    def stack(self, ctx_full):
        """Return counts from order 0 up to order-k (each length-4 array)."""
        out = []
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])  # last o symbols
            out.append(self.tables[o].get(ctx, np.zeros(4, dtype=np.int64)))
        return out

    def probs(self, ctx_full):
        return make_probs(self.stack(ctx_full), self.alpha)

    def update(self, ctx_full, sym):
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])
            arr = self.tables[o].get(ctx)
            if arr is None:
                arr = np.zeros(4, dtype=np.int64)
                self.tables[o][ctx] = arr
            arr[sym] += 1

# ---- encode / decode ----
def encode_adaptive(ids, k, alpha=1.0, side=None):
    """side (optional): integer array same length as ids, extending ctx."""
    enc = RangeEncoder()
    model = AdaptiveModel(k, alpha)
    n = len(ids)
    pad = np.zeros(k, dtype=np.int32)
    for i in range(n):
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        if side is not None:
            # Use side[i] as an additional "super-context" key by hashing into a
            # separate bank of tables via a wrapped AdaptiveModel-per-bucket.
            raise NotImplementedError("Use AdaptivePolarModel instead")
        p = model.probs(ctx)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]))
    return bytes(enc.get_compressed().tobytes())

def decode_adaptive(payload, n, k, alpha=1.0):
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    # pad to multiple of 4 bytes for uint32 view
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
        s = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
        out[i] = s
        model.update(ctx, s)
    return out

# =================================================================
# POLAR-CONDITIONED ADAPTIVE MODEL
# =================================================================
# Observation from V4: polar centroid carries info, but static conditional
# tables have catastrophic overhead. Adaptive tables inherit zero overhead
# but still specialize per-bucket. Let's test whether specialization is
# worth the statistical fragmentation.

class AdaptivePolarModel:
    def __init__(self, k, n_buckets, alpha=1.0):
        self.k = k
        self.n_buckets = n_buckets
        self.alpha = alpha
        # per-bucket AdaptiveModel
        self.banks = [AdaptiveModel(k, alpha) for _ in range(n_buckets)]
    def probs(self, ctx_full, b):
        return self.banks[b].probs(ctx_full)
    def update(self, ctx_full, sym, b):
        self.banks[b].update(ctx_full, sym)

def block_centroid_buckets(ids, block_size=64, n_buckets=8):
    """Per-base bucket id derived from running block centroids.
       Uses CAUSAL (past-only) windows so decoder can reconstruct."""
    z = np.exp(1j * PHASES[ids])
    buckets = np.zeros(len(ids), dtype=np.int32)
    s = 0 + 0j
    for i in range(len(ids)):
        if i < block_size:
            window = z[:i+1]
        else:
            window = z[i-block_size+1:i+1]
        c = window.sum() / len(window)
        ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
        buckets[i] = int(ang * n_buckets / (2*np.pi)) % n_buckets
    return buckets

def encode_adaptive_polar(ids, k, block_size=64, n_buckets=8, alpha=1.0):
    enc = RangeEncoder()
    model = AdaptivePolarModel(k, n_buckets, alpha)
    # decoder will rebuild buckets causally from its own reconstructed ids
    n = len(ids)
    pad = np.zeros(k, dtype=np.int32)
    z = np.exp(1j * PHASES[ids])
    s_complex = 0 + 0j
    window = []
    for i in range(n):
        window.append(z[i])
        if len(window) > block_size: window.pop(0)
        c = sum(window) / len(window)
        ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
        b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        # bucket at step i uses symbols 0..i INCLUSIVE.
        # For decoder to reproduce, it must know bucket of step i BEFORE decoding.
        # So we must use CAUSAL-PAST-ONLY buckets: compute on window of 0..i-1.
        pass
    # Redo with causal-past-only
    enc = RangeEncoder()
    model = AdaptivePolarModel(k, n_buckets, alpha)
    window = []
    for i in range(n):
        if not window:
            b = 0
        else:
            c = sum(window) / len(window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        p = model.probs(ctx, b)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]), b)
        window.append(z[i])
        if len(window) > block_size: window.pop(0)
    return bytes(enc.get_compressed().tobytes())

def decode_adaptive_polar(payload, n, k, block_size=64, n_buckets=8, alpha=1.0):
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    pad_bytes = (-len(arr)) % 4
    if pad_bytes: arr = np.concatenate([arr, np.zeros(pad_bytes, dtype=np.uint8)])
    u32 = arr.view(np.uint32).copy()
    dec = RangeDecoder(u32)
    model = AdaptivePolarModel(k, n_buckets, alpha)
    out = np.zeros(n, dtype=np.int32)
    window = []
    pad = np.zeros(k, dtype=np.int32)
    for i in range(n):
        if not window:
            b = 0
        else:
            c = sum(window) / len(window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        ctx = np.concatenate([pad, out[:i]])[-k:]
        p = model.probs(ctx, b)
        sym = dec.decode(Categorical(p, perfect=False))
        s = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
        out[i] = s
        model.update(ctx, s, b)
        window.append(np.exp(1j * PHASES[s]))
        if len(window) > block_size: window.pop(0)
    return out

# =================================================================
# FILE-LEVEL POLARQUANT CODEC (LOSSLESS FOR FASTA / CSV / ANY BYTES)
# =================================================================
#
# Strategy:
#   1) Convert raw bytes to a DNA-alphabet stream (A/C/G/T = 2 bits).
#   2) Compress that stream with V5-plain or V5-polar arithmetic coding.
#   3) Store a small JSON header + payload in a .pqc container.

def bytes_to_ids(raw_bytes):
    """Map each byte to 4 DNA ids (2 bits per symbol), MSB first."""
    b = np.frombuffer(raw_bytes, dtype=np.uint8)
    out = np.empty(len(b) * 4, dtype=np.int32)
    out[0::4] = (b >> 6) & 0b11
    out[1::4] = (b >> 4) & 0b11
    out[2::4] = (b >> 2) & 0b11
    out[3::4] = b & 0b11
    return out

def ids_to_bytes(ids, n_bytes):
    """Inverse of bytes_to_ids."""
    if len(ids) != n_bytes * 4:
        raise ValueError("Corrupt stream: DNA id length does not match original byte length.")
    b0 = (ids[0::4].astype(np.uint8) & 0b11) << 6
    b1 = (ids[1::4].astype(np.uint8) & 0b11) << 4
    b2 = (ids[2::4].astype(np.uint8) & 0b11) << 2
    b3 = (ids[3::4].astype(np.uint8) & 0b11)
    return bytes((b0 | b1 | b2 | b3).tolist())

def compress_file(input_path, output_path=None, mode="polar", k=4, block_size=64, n_buckets=8, alpha=1.0):
    src = Path(input_path)
    if not src.exists():
        raise FileNotFoundError(f"Input file not found: {src}")
    raw = src.read_bytes()
    ids = bytes_to_ids(raw)
    n = int(len(ids))

    if mode == "plain":
        payload = encode_adaptive(ids, k=k, alpha=alpha)
    elif mode == "polar":
        payload = encode_adaptive_polar(ids, k=k, block_size=block_size, n_buckets=n_buckets, alpha=alpha)
    else:
        raise ValueError("mode must be 'plain' or 'polar'")

    header = {
        "version": 1,
        "mode": mode,
        "k": int(k),
        "alpha": float(alpha),
        "block_size": int(block_size),
        "n_buckets": int(n_buckets),
        "n_bytes": int(len(raw)),
        "n_ids": n,
        "source_name": src.name,
    }
    header_b = json.dumps(header, separators=(",", ":")).encode("utf-8")
    out = Path(output_path) if output_path else src.with_suffix(src.suffix + ".pqc")
    out.write_bytes(MAGIC + struct.pack("<I", len(header_b)) + header_b + payload)

    return {
        "input": str(src),
        "output": str(out),
        "raw_bytes": len(raw),
        "compressed_bytes": out.stat().st_size,
    }

def decompress_file(input_path, output_path=None):
    src = Path(input_path)
    blob = src.read_bytes()
    if len(blob) < 8 or blob[:4] != MAGIC:
        raise ValueError("Not a PolarQuant .pqc file or corrupt header.")
    hlen = struct.unpack("<I", blob[4:8])[0]
    h0 = 8
    h1 = h0 + hlen
    header = json.loads(blob[h0:h1].decode("utf-8"))
    payload = blob[h1:]

    mode = header["mode"]
    n_ids = int(header["n_ids"])
    n_bytes = int(header["n_bytes"])
    k = int(header["k"])
    alpha = float(header["alpha"])
    block_size = int(header["block_size"])
    n_buckets = int(header["n_buckets"])

    if mode == "plain":
        ids = decode_adaptive(payload, n_ids, k=k, alpha=alpha)
    elif mode == "polar":
        ids = decode_adaptive_polar(payload, n_ids, k=k, block_size=block_size, n_buckets=n_buckets, alpha=alpha)
    else:
        raise ValueError(f"Unsupported mode in archive: {mode}")

    raw = ids_to_bytes(ids, n_bytes)
    if output_path:
        out = Path(output_path)
    else:
        # default: strip trailing .pqc if present
        out = src.with_suffix("") if src.suffix == ".pqc" else src.with_name(src.name + ".out")
    out.write_bytes(raw)
    return {"input": str(src), "output": str(out), "bytes": len(raw)}

# =================================================================
# BENCHMARK
# =================================================================
def bench(seq, label, k=4):
    ids = seq_to_ids(seq)
    n = len(seq)
    raw = seq.encode()
    base_2bit = (n + 3) // 4

    print(f"\n{'='*78}")
    print(f"{label}  (n = {n:,} bases,  k = {k})")
    print(f"{'='*78}")
    print(f"{'Method':<42}{'Bytes':>10}{'Bits/base':>12}{'vs 2-bit':>10}")
    print("-" * 78)
    def row(name, b):
        print(f"{name:<42}{b:>10}{b*8/n:>12.3f}{base_2bit/b:>9.2f}x")
    row("2-bit packed baseline", base_2bit)
    row("Raw ASCII + bz2", len(bz2.compress(raw)))
    row("Raw ASCII + lzma", len(lzma.compress(raw)))

    # V5-plain
    payload = encode_adaptive(ids, k=k)
    row(f"V5-plain adaptive (k={k})", len(payload))

    # V5-plain at higher k (feasible now that overhead is zero)
    payload6 = encode_adaptive(ids, k=6)
    row("V5-plain adaptive (k=6)", len(payload6))

    # V5-polar adaptive, several bucket counts
    for nb in (4, 8, 16):
        pp = encode_adaptive_polar(ids, k=k, block_size=64, n_buckets=nb)
        row(f"V5-polar adaptive (k={k}, buckets={nb})", len(pp))

def roundtrip(label, seq, k=4):
    ids = seq_to_ids(seq)
    p = encode_adaptive(ids, k=k)
    d = decode_adaptive(p, len(ids), k=k)
    ok = np.array_equal(ids, d)
    p2 = encode_adaptive_polar(ids, k=k, block_size=64, n_buckets=8)
    d2 = decode_adaptive_polar(p2, len(ids), k=k, block_size=64, n_buckets=8)
    ok2 = np.array_equal(ids, d2)
    print(f"Round-trip {label}: V5-plain={'OK' if ok else 'FAIL'}  V5-polar={'OK' if ok2 else 'FAIL'}")

def main():
    parser = argparse.ArgumentParser(description="PolarQuant V5 compressor for DNA and file bytes.")
    sub = parser.add_subparsers(dest="cmd")

    p_bench = sub.add_parser("bench", help="Run built-in synthetic DNA benchmarks.")
    p_bench.add_argument("--k", type=int, default=4)
    p_bench.add_argument("--n", type=int, default=50_000)

    p_c = sub.add_parser("compress", help="Compress a file to .pqc using PolarQuant strategy.")
    p_c.add_argument("input", help="Input file (e.g., .fasta, .csv).")
    p_c.add_argument("-o", "--output", default=None, help="Output .pqc path.")
    p_c.add_argument("--mode", choices=["plain", "polar"], default="polar")
    p_c.add_argument("--k", type=int, default=4)
    p_c.add_argument("--block-size", type=int, default=64)
    p_c.add_argument("--buckets", type=int, default=8)
    p_c.add_argument("--alpha", type=float, default=1.0)

    p_d = sub.add_parser("decompress", help="Decompress a .pqc archive.")
    p_d.add_argument("input", help="Input .pqc path.")
    p_d.add_argument("-o", "--output", default=None, help="Recovered output file path.")

    args = parser.parse_args()

    if args.cmd == "compress":
        stats = compress_file(
            input_path=args.input,
            output_path=args.output,
            mode=args.mode,
            k=args.k,
            block_size=args.block_size,
            n_buckets=args.buckets,
            alpha=args.alpha,
        )
        ratio = stats["raw_bytes"] / stats["compressed_bytes"] if stats["compressed_bytes"] else 0.0
        print(f"Compressed: {stats['input']} -> {stats['output']}")
        print(f"Size: {stats['raw_bytes']:,} -> {stats['compressed_bytes']:,} bytes ({ratio:.2f}x)")
        return

    if args.cmd == "decompress":
        stats = decompress_file(args.input, args.output)
        print(f"Decompressed: {stats['input']} -> {stats['output']} ({stats['bytes']:,} bytes)")
        return

    # Default behavior: benchmark if no subcommand provided.
    k = 4 if args.cmd is None else args.k
    N = 50_000 if args.cmd is None else args.n
    roundtrip("random 1k", random_dna(1000, seed=7), k=k)
    roundtrip("AT-rich 1k", skewed_dna(1000, seed=8), k=k)
    roundtrip("mtDNA 1k",  MTDNA_TILE[:1000], k=k)
    bench(random_dna(N),        "RANDOM DNA",        k=k)
    bench(skewed_dna(N),        "AT-RICH DNA",       k=k)
    bench(repeat_dna(N),        "REPETITIVE DNA",    k=k)
    bench(MTDNA_TILE[:N],       "HUMAN mtDNA (tiled)", k=k)

if __name__ == "__main__":
    main()
