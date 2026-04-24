"""
polar_v5_fastq.py  --  Extract FASTQ from zip, compress with V5 block-parallel.

Extracts DNA sequences from all .fastq files in fastqfiles.zip,
cleans non-ACGT characters, and benchmarks:
  - 2-bit packed baseline
  - bz2, lzma
  - V5 monolithic (k=4, k=6)
  - V5 block-parallel at tile=4096
  - V5-polar block-parallel at tile=4096, B=16

Reports per-file and aggregate stats.
"""

import numpy as np, random, zlib, bz2, lzma, struct, time, zipfile, re, os, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import constriction

RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical  = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
I2B = {i: b for i, b in enumerate(BASES)}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])

# ================================================================
# FASTQ EXTRACTION
# ================================================================
def extract_fastq_sequences(zippath, max_bases_per_file=None):
    """Extract DNA sequences from all .fastq files in a zip.
    Returns dict: filename -> concatenated sequence string (ACGT only)."""
    results = {}
    with zipfile.ZipFile(zippath) as z:
        for info in z.infolist():
            if not info.filename.endswith('.fastq'):
                continue
            seqs = []
            total_bases = 0
            with z.open(info.filename) as f:
                line_num = 0
                for raw_line in f:
                    line_num += 1
                    # FASTQ: line 1=header(@), 2=seq, 3=sep(+), 4=quality
                    if line_num % 4 == 2:  # sequence line
                        seq = raw_line.decode().strip().upper()
                        # Keep only ACGT, drop N and other ambiguity codes
                        cleaned = re.sub(r'[^ACGT]', '', seq)
                        if cleaned:
                            seqs.append(cleaned)
                            total_bases += len(cleaned)
                            if max_bases_per_file and total_bases >= max_bases_per_file:
                                break
            full_seq = "".join(seqs)
            if max_bases_per_file:
                full_seq = full_seq[:max_bases_per_file]
            results[info.filename] = full_seq
            print(f"  Extracted {info.filename}: {len(full_seq):,} bases "
                  f"({len(seqs):,} reads)")
    return results

# ================================================================
# ADAPTIVE PPM MODEL (from V5)
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

class AdaptivePolarModel:
    def __init__(self, k, n_buckets, alpha=1.0):
        self.k = k; self.n_buckets = n_buckets; self.alpha = alpha
        self.banks = [AdaptiveModel(k, alpha) for _ in range(n_buckets)]
    def probs(self, ctx, b): return self.banks[b].probs(ctx)
    def update(self, ctx, sym, b): self.banks[b].update(ctx, sym)

# ================================================================
# TILE ENCODE / DECODE
# ================================================================
def seq_to_ids(s):
    return np.fromiter((B2I[c] for c in s), dtype=np.int32, count=len(s))

def ids_to_seq(ids):
    return "".join(I2B[int(i)] for i in ids)

def encode_tile(ids, k=4, alpha=1.0):
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

def encode_tile_polar(ids, k=4, block_size=64, n_buckets=16, alpha=1.0):
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

# ================================================================
# BLOCK-PARALLEL CODEC
# ================================================================
def compress_parallel(seq, tile_size, k=4, polar=False, n_buckets=16,
                      max_workers=4):
    ids = seq_to_ids(seq)
    n = len(ids)
    n_tiles = (n + tile_size - 1) // tile_size
    tiles = [ids[i*tile_size : (i+1)*tile_size] for i in range(n_tiles)]
    enc_fn = (lambda t: encode_tile_polar(t, k, 64, n_buckets)) if polar \
             else (lambda t: encode_tile(t, k))
    payloads = [None] * n_tiles
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(enc_fn, t): i for i, t in enumerate(tiles)}
        for f in as_completed(futs):
            payloads[futs[f]] = f.result()
    # Pack
    parts = [struct.pack("<I", n_tiles)]
    for i in range(n_tiles):
        parts.append(struct.pack("<II", len(tiles[i]), len(payloads[i])))
    for p in payloads:
        parts.append(p)
    return b"".join(parts)

def decompress_parallel(data, k=4, polar=False, n_buckets=16, max_workers=4):
    off = 0
    n_tiles = struct.unpack_from("<I", data, off)[0]; off += 4
    tile_meta = []
    for _ in range(n_tiles):
        tlen, plen = struct.unpack_from("<II", data, off); off += 8
        tile_meta.append((tlen, plen))
    dec_fn = (lambda pay, tl: decode_tile(pay, tl, k)) if not polar \
             else (lambda pay, tl: decode_tile(pay, tl, k))  # plain decode for verify
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
# BENCHMARK ONE FILE
# ================================================================
def bench_one(seq, label, k=4, tile_size=4096):
    n = len(seq)
    raw = seq.encode()
    base_2bit = (n + 3) // 4
    results = {}

    print(f"\n{'='*90}")
    print(f"{label}  ({n:,} bases)")
    print(f"{'='*90}")
    print(f"{'Method':<52}{'Bytes':>10}{'bpb':>8}{'vs 2bit':>9}{'Time':>8}")
    print("-" * 90)

    def row(name, b, t=0):
        ts = f"{t:.1f}s" if t else ""
        r = base_2bit / max(b, 1)
        results[name] = (b, b*8/n, r)
        print(f"{name:<52}{b:>10}{b*8/n:>8.3f}{r:>8.1f}x{ts:>8}")

    row("2-bit packed", base_2bit)
    row("raw + bz2", len(bz2.compress(raw)))
    row("raw + lzma", len(lzma.compress(raw)))

    # 2-bit packed + lzma (the smart baseline)
    packed = bytearray((n + 3) // 4)
    ids = seq_to_ids(seq)
    for i, s in enumerate(ids):
        packed[i // 4] |= (s << (2 * (i % 4)))
    row("2-bit + lzma", len(lzma.compress(bytes(packed))))
    row("2-bit + bz2", len(bz2.compress(bytes(packed))))

    # V5 monolithic (only if small enough)
    if n <= 20000:
        t0 = time.time()
        mono = encode_tile(ids, k=k)
        row(f"V5 mono (k={k})", len(mono), time.time()-t0)

    # V5 block-parallel plain
    n_tiles = (n + tile_size - 1) // tile_size
    t0 = time.time()
    blob = compress_parallel(seq, tile_size, k=k, polar=False)
    row(f"V5-parallel plain (tile={tile_size}, {n_tiles} tiles)", len(blob),
        time.time()-t0)

    # V5 block-parallel polar
    t0 = time.time()
    blob_p = compress_parallel(seq, tile_size, k=k, polar=True, n_buckets=16)
    row(f"V5-parallel polar (tile={tile_size}, B=16)", len(blob_p),
        time.time()-t0)

    return results

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    ZIP = "fastqfiles.zip"
    # Use first 20k bases per file for speed (Python is slow)
    MAX_BASES = 20_000
    TILE = 4096
    K = 4

    print("=" * 90)
    print("POLAR DNA COMPRESSOR V5  --  FASTQ BENCHMARK")
    print(f"Max bases/file: {MAX_BASES:,}   Tile size: {TILE}   Context order: {K}")
    print("=" * 90)

    print("\nExtracting FASTQ files from zip...")
    files = extract_fastq_sequences(ZIP, max_bases_per_file=MAX_BASES)

    print(f"\nExtracted {len(files)} files, "
          f"{sum(len(v) for v in files.values()):,} total bases")

    # Round-trip check on first file
    first_name = sorted(files.keys())[0]
    first_seq = files[first_name][:2000]
    blob = compress_parallel(first_seq, 256, k=K, polar=False)
    dec = decompress_parallel(blob, k=K, polar=False)
    print(f"\nRound-trip check ({first_name[:30]}..., 2k bases): "
          f"{'OK' if dec == first_seq else 'FAIL'}")

    # Benchmark a representative sample (pick 5 diverse files)
    sorted_files = sorted(files.items(), key=lambda x: len(x[1]), reverse=True)
    # Pick: largest, smallest, and 3 evenly spaced
    picks = [sorted_files[0], sorted_files[-1],
             sorted_files[len(sorted_files)//4],
             sorted_files[len(sorted_files)//2],
             sorted_files[3*len(sorted_files)//4]]
    seen = set()
    unique_picks = []
    for name, seq in picks:
        if name not in seen:
            seen.add(name)
            unique_picks.append((name, seq))

    all_results = {}
    for name, seq in unique_picks:
        short = name.replace('_2019_minq7.fastq', '')
        all_results[short] = bench_one(seq, short, k=K, tile_size=TILE)

    # Aggregate: concatenate ALL file sequences and bench
    print("\n\n" + "#" * 90)
    print("AGGREGATE: all files concatenated")
    print("#" * 90)
    all_seq = "".join(files[k] for k in sorted(files.keys()))
    bench_one(all_seq, f"ALL {len(files)} FILES CONCATENATED", k=K, tile_size=TILE)

    # Summary table
    print(f"\n\n{'='*70}")
    print("SUMMARY: bits/base across methods")
    print(f"{'='*70}")
    print(f"{'File':<30}{'2bit':>7}{'bz2':>7}{'lzma':>7}{'V5par':>7}{'V5pol':>7}")
    print("-" * 70)
    for name, res in all_results.items():
        short = name[:28]
        bpb_2bit = res.get("2-bit packed", (0,2.0,0))[1]
        bpb_bz2  = res.get("raw + bz2", (0,0,0))[1]
        bpb_lzma = res.get("raw + lzma", (0,0,0))[1]
        bpb_par  = [v[1] for k,v in res.items() if "V5-parallel plain" in k]
        bpb_pol  = [v[1] for k,v in res.items() if "V5-parallel polar" in k]
        print(f"{short:<30}{bpb_2bit:>7.3f}{bpb_bz2:>7.3f}{bpb_lzma:>7.3f}"
              f"{bpb_par[0] if bpb_par else 0:>7.3f}"
              f"{bpb_pol[0] if bpb_pol else 0:>7.3f}")
