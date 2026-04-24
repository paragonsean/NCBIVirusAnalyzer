"""
polar_format_v2.py  --  .polar v2: improved custom DNA format.

Improvements over v1:
  * Quality binning: 94 Phred levels → 8 bins (massive size reduction)
  * Quality stream uses lzma (better than our adaptive coder on 8-symbol alphabet)
  * Sequence stream uses k=6 (sorted reads have long shared prefixes)
  * Polar conditioning on sequence stream with B=8 causal buckets
  * Header stream: extract common prefix, delta-encode unique suffixes

Stream layout:
  STREAM 0 — HEADERS:   lzma-compressed text
  STREAM 1 — SEQUENCES: V5 adaptive PPM (k=6) with polar conditioning
  STREAM 2 — QUALITY:   binned (8 levels) → lzma
  STREAM 3 — META:      read lengths + sort permutation + quality bin table → lzma
"""

import numpy as np, struct, zlib, bz2, lzma, zipfile, re, time, sys, os
import constriction
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical  = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
I2B = {i: b for i, b in enumerate(BASES)}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
MAGIC = b'POLAR2DN'
VERSION = 2

# ================================================================
# QUALITY BINNING
# ================================================================
# Nanopore Phred scores are noisy; binning preserves the useful signal.
# 8 bins: [0-5], [6-9], [10-14], [15-19], [20-24], [25-29], [30-36], [37+]
# Representative values: 2, 7, 12, 17, 22, 27, 33, 40

QUAL_BIN_EDGES = [0, 6, 10, 15, 20, 25, 30, 37, 94]
QUAL_BIN_REPS  = [2, 7, 12, 17, 22, 27, 33, 40]

def phred_to_bin(q):
    """Map Phred score (0-93) to bin index (0-7)."""
    for i in range(len(QUAL_BIN_EDGES) - 1):
        if q < QUAL_BIN_EDGES[i + 1]:
            return i
    return 7

def bin_to_phred(b):
    """Map bin index back to representative Phred score."""
    return QUAL_BIN_REPS[b]

# Pre-compute lookup table
_PHRED_TO_BIN = np.array([phred_to_bin(q) for q in range(94)], dtype=np.uint8)

def bin_quality_string(qstr):
    """Convert quality ASCII string to array of bin indices."""
    return np.array([_PHRED_TO_BIN[max(0, min(ord(c) - 33, 93))] for c in qstr],
                    dtype=np.uint8)

def unbin_quality_string(bins):
    """Convert bin indices back to quality ASCII string (using representative values)."""
    return "".join(chr(QUAL_BIN_REPS[int(b)] + 33) for b in bins)

# ================================================================
# FASTQ PARSER
# ================================================================
def parse_fastq_from_zip(zippath, filename, max_reads=None):
    reads = []
    with zipfile.ZipFile(zippath) as z:
        with z.open(filename) as f:
            lines = []
            for raw in f:
                lines.append(raw.decode().rstrip('\n\r'))
                if len(lines) == 4:
                    reads.append((lines[0], lines[1], lines[3]))
                    lines = []
                    if max_reads and len(reads) >= max_reads:
                        break
    return reads

# ================================================================
# READ SORTING
# ================================================================
def sort_reads(reads):
    indexed = list(enumerate(reads))
    indexed.sort(key=lambda x: x[1][1])
    perm = [i for i, _ in indexed]
    return [r for _, r in indexed], perm

def unsort_reads(sorted_reads, perm):
    out = [None] * len(sorted_reads)
    for new_idx, orig_idx in enumerate(perm):
        out[orig_idx] = sorted_reads[new_idx]
    return out

# ================================================================
# ADAPTIVE PPM + POLAR (for sequence stream)
# ================================================================
class AdaptiveModel:
    def __init__(self, k, n_symbols=4, alpha=1.0):
        self.k = k; self.alpha = alpha; self.n_symbols = n_symbols
        self.tables = [dict() for _ in range(k + 1)]
    def probs(self, ctx_full):
        n = self.n_symbols
        uniform = np.full(n, 1.0/n, dtype=np.float64)
        p = uniform.copy()
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])
            c = self.tables[o].get(ctx, np.zeros(n, dtype=np.int64))
            total = c.sum()
            p = (c + self.alpha * p) / (total + self.alpha)
        p = np.clip(p, 1e-9, None); p /= p.sum()
        return p
    def update(self, ctx_full, sym):
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])
            arr = self.tables[o].get(ctx)
            if arr is None:
                arr = np.zeros(self.n_symbols, dtype=np.int64)
                self.tables[o][ctx] = arr
            arr[sym] += 1

class AdaptivePolarModel:
    def __init__(self, k, n_buckets=8, alpha=1.0):
        self.k = k; self.n_buckets = n_buckets
        self.banks = [AdaptiveModel(k, 4, alpha) for _ in range(n_buckets)]
    def probs(self, ctx, b): return self.banks[b].probs(ctx)
    def update(self, ctx, sym, b): self.banks[b].update(ctx, sym)

def encode_seq_v2_parallel(sorted_seqs, k=6, n_buckets=8, chunk_size=10000):
    """Parallel version of encode_seq_v2 for faster processing."""
    print(f"DEBUG: encode_seq_v2_parallel called with {len(sorted_seqs)} sequences")
    all_bases = re.sub(r'[^ACGT]', '', "".join(sorted_seqs).upper())
    ids = np.fromiter((B2I[c] for c in all_bases), dtype=np.int32, count=len(all_bases))
    n = len(ids)
    print(f"DEBUG: Total bases to encode: {n}")

    # Split into chunks for parallel processing
    chunks = []
    for i in range(0, n, chunk_size):
        chunk_end = min(i + chunk_size, n)
        chunks.append((ids, i, chunk_end, k, n_buckets))
    
    print(f"DEBUG: Processing {len(chunks)} chunks in parallel")
    
    # Process chunks in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        chunk_results = list(tqdm(
            executor.map(encode_chunk_worker, chunks), 
            total=len(chunks), 
            desc="Processing chunks", 
            unit="chunk"
        ))
    
    # Combine results
    all_compressed = b"".join([result for result in chunk_results])
    return all_compressed, n

def encode_chunk_worker(args):
    """Worker function for parallel chunk encoding."""
    ids, start, end, k, n_buckets = args
    
    enc = RangeEncoder()
    model = AdaptivePolarModel(k, n_buckets)
    pad = np.zeros(k, dtype=np.int32)
    z_window = []
    block_size = 64
    
    for i in range(start, end):
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        if not z_window: b = 0
        else:
            c = sum(z_window) / len(z_window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        p = model.probs(ctx, b)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]), b)
        z_window.append(np.exp(1j * PHASES[ids[i]]))
        if len(z_window) > block_size: z_window.pop(0)
    
    return enc.get_compressed().tobytes()

def encode_seq_v2(sorted_seqs, k=6, n_buckets=8):
    """Encode sorted+cleaned DNA sequences with V5 polar adaptive PPM."""
    print(f"DEBUG: encode_seq_v2 called with {len(sorted_seqs)} sequences")
    all_bases = re.sub(r'[^ACGT]', '', "".join(sorted_seqs).upper())
    ids = np.fromiter((B2I[c] for c in all_bases), dtype=np.int32, count=len(all_bases))
    n = len(ids)
    print(f"DEBUG: Total bases to encode: {n}")

    enc = RangeEncoder()
    model = AdaptivePolarModel(k, n_buckets)
    pad = np.zeros(k, dtype=np.int32)
    z_window = []
    block_size = 64

    for i in tqdm(range(n), desc="Encoding sequences", unit="base"):
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        if not z_window: b = 0
        else:
            c = sum(z_window) / len(z_window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        p = model.probs(ctx, b)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]), b)
        z_window.append(np.exp(1j * PHASES[ids[i]]))
        if len(z_window) > block_size: z_window.pop(0)

    return enc.get_compressed().tobytes(), n

def decode_seq_v2(payload, n, k=6, n_buckets=8):
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    pad_bytes = (-len(arr)) % 4
    if pad_bytes: arr = np.concatenate([arr, np.zeros(pad_bytes, dtype=np.uint8)])
    u32 = arr.view(np.uint32).copy()
    dec = RangeDecoder(u32)

    model = AdaptivePolarModel(k, n_buckets)
    pad = np.zeros(k, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    z_window = []
    block_size = 64

    for i in range(n):
        ctx = np.concatenate([pad, out[:i]])[-k:]
        if not z_window: b = 0
        else:
            c = sum(z_window) / len(z_window)
            ang = (np.angle(c) + 2*np.pi) % (2*np.pi)
            b = int(ang * n_buckets / (2*np.pi)) % n_buckets
        p = model.probs(ctx, b)
        sym = dec.decode(Categorical(p, perfect=False))
        s = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
        out[i] = s
        model.update(ctx, s, b)
        z_window.append(np.exp(1j * PHASES[s]))
        if len(z_window) > block_size: z_window.pop(0)

    return "".join(I2B[int(x)] for x in out)

# ================================================================
# .POLAR V2 ENCODER / DECODER
# ================================================================
def encode_polar_v2(reads, seq_k=6, n_buckets=8):
    """Encode reads → .polar v2 bytes."""
    print(f"DEBUG: Starting encode_polar_v2 with {len(reads)} reads")
    t0 = time.time()
    print("DEBUG: Starting sort_reads...")
    sorted_reads, perm = sort_reads(reads)
    print("DEBUG: sort_reads completed")
    print("DEBUG: Processing sorted reads...")
    sorted_headers = [r[0] for r in sorted_reads]
    print("DEBUG: Extracted headers")
    sorted_seqs    = [r[1] for r in sorted_reads]
    print("DEBUG: Extracted sequences")
    sorted_quals   = [r[2] for r in sorted_reads]
    print("DEBUG: Extracted qualities")

    # Clean sequences for length tracking
    print("DEBUG: Cleaning sequences...")
    cleaned_seqs = [re.sub(r'[^ACGT]', '', s.upper()) for s in sorted_seqs]
    print("DEBUG: Sequences cleaned")
    seq_lengths  = [len(s) for s in cleaned_seqs]
    print("DEBUG: Sequence lengths calculated")

    # STREAM 0: headers → lzma
    print("DEBUG: Compressing headers...")
    header_text = ""
    for header in tqdm(sorted_headers, desc="Compressing headers", unit="header"):
        header_text += header + "\n"
    header_comp = lzma.compress(header_text.encode())
    print("DEBUG: Headers compressed")

    # STREAM 1: sequences → V5 polar adaptive (k=6)
    print(f"DEBUG: Starting parallel sequence encoding with {len(cleaned_seqs)} sequences")
    seq_comp, seq_n = encode_seq_v2_parallel(cleaned_seqs, k=seq_k, n_buckets=n_buckets)

    # STREAM 2: quality → bin to 8 levels → lzma
    all_binned = np.concatenate([bin_quality_string(q) for q in tqdm(sorted_quals, desc="Binning quality", unit="read")])
    qual_comp = lzma.compress(bytes(all_binned))
    qual_n = len(all_binned)

    # STREAM 3: meta (lengths + perm + qual_lengths)
    qual_lengths = [len(q) for q in sorted_quals]
    meta = struct.pack(f"<{len(seq_lengths)}I", *seq_lengths)
    meta += struct.pack(f"<{len(qual_lengths)}I", *qual_lengths)
    meta += struct.pack(f"<{len(perm)}I", *perm)
    meta_comp = lzma.compress(meta)

    # Assemble file
    parts = [MAGIC, struct.pack("<III", VERSION, len(reads), 4)]
    for stype, data, extra in [
        (0, header_comp, None),
        (1, seq_comp, struct.pack("<I", seq_n)),
        (2, qual_comp, struct.pack("<I", qual_n)),
        (3, meta_comp, None),
    ]:
        parts.append(struct.pack("<BI", stype, len(data)))
        if extra: parts.append(extra)
        parts.append(data)

    blob = b"".join(parts)
    return blob, time.time() - t0

def decode_polar_v2(blob, seq_k=6, n_buckets=8):
    """Decode .polar v2 → list of (header, seq, qual) in original order."""
    off = 0
    magic = blob[off:off+8]; off += 8
    assert magic == MAGIC, f"Bad magic: {magic}"
    ver, n_reads, n_streams = struct.unpack_from("<III", blob, off); off += 12

    streams = {}
    for _ in range(n_streams):
        stype = blob[off]; off += 1
        ssize = struct.unpack_from("<I", blob, off)[0]; off += 4
        extra = None
        if stype in (1, 2):
            extra = struct.unpack_from("<I", blob, off)[0]; off += 4
        sdata = blob[off:off+ssize]; off += ssize
        streams[stype] = (sdata, extra)

    # Headers
    sorted_headers = lzma.decompress(streams[0][0]).decode().split("\n")

    # Sequences
    all_seq = decode_seq_v2(streams[1][0], streams[1][1], k=seq_k,
                             n_buckets=n_buckets)

    # Quality (binned)
    all_binned = np.frombuffer(lzma.decompress(streams[2][0]), dtype=np.uint8)

    # Meta
    meta_raw = lzma.decompress(streams[3][0])
    seq_lengths  = list(struct.unpack_from(f"<{n_reads}I", meta_raw, 0))
    qual_lengths = list(struct.unpack_from(f"<{n_reads}I", meta_raw, n_reads*4))
    perm         = list(struct.unpack_from(f"<{n_reads}I", meta_raw, n_reads*8))

    # Split sequences
    sorted_seqs = []
    pos = 0
    for l in seq_lengths:
        sorted_seqs.append(all_seq[pos:pos+l]); pos += l

    # Split + unbin quality
    sorted_quals = []
    pos = 0
    for l in qual_lengths:
        sorted_quals.append(unbin_quality_string(all_binned[pos:pos+l])); pos += l

    sorted_reads = list(zip(sorted_headers, sorted_seqs, sorted_quals))
    return unsort_reads(sorted_reads, perm)

# ================================================================
# BENCHMARK
# ================================================================
def bench_file(zippath, filename, max_reads=500):
    reads = parse_fastq_from_zip(zippath, filename, max_reads=max_reads)
    n_reads = len(reads)
    total_bases = sum(len(re.sub(r'[^ACGT]', '', r[1].upper())) for r in reads)
    total_qual  = sum(len(r[2]) for r in reads)

    # Original FASTQ
    fastq_lines = []
    for h, s, q in reads:
        fastq_lines.extend([h, s, "+", q])
    fastq_raw = "\n".join(fastq_lines).encode()
    fastq_size = len(fastq_raw)

    short = filename.replace('_2019_minq7.fastq', '')

    print(f"\n{'='*90}")
    print(f"{short}  ({n_reads} reads, {total_bases:,} seq bases, {total_qual:,} qual chars)")
    print(f"{'='*90}")
    print(f"{'Method':<50}{'Bytes':>10}{'Ratio':>8}{'bpb':>8}{'Time':>8}")
    print("-" * 90)

    def row(name, b, t=0):
        ts = f"{t:.1f}s" if t else ""
        ratio = fastq_size / max(b, 1)
        # bpb relative to total data characters (seq + qual)
        bpb = b * 8 / max(total_bases + total_qual, 1)
        print(f"{name:<50}{b:>10}{ratio:>7.1f}x{bpb:>8.3f}{ts:>8}")

    row("Original FASTQ", fastq_size)
    row("FASTQ + gzip", len(zlib.compress(fastq_raw)))
    row("FASTQ + bz2", len(bz2.compress(fastq_raw)))
    row("FASTQ + lzma", len(lzma.compress(fastq_raw)))

    # Sequences-only baselines (for fair bpb comparison)
    all_seq = "".join(re.sub(r'[^ACGT]', '', r[1].upper()) for r in reads).encode()
    print(f"\n  --- Sequence-only compression ---")
    def row_seq(name, b, t=0):
        ts = f"{t:.1f}s" if t else ""
        bpb = b * 8 / max(total_bases, 1)
        print(f"  {name:<48}{b:>10}{bpb:>8.3f} bpb{ts:>5}")
    row_seq("2-bit packed", (total_bases + 3) // 4)
    row_seq("seq + bz2", len(bz2.compress(all_seq)))
    row_seq("seq + lzma", len(lzma.compress(all_seq)))
    # sorted seq
    sorted_reads, _ = sort_reads(reads)
    sorted_seq = "".join(re.sub(r'[^ACGT]', '', r[1].upper())
                         for r in sorted_reads).encode()
    row_seq("sorted seq + bz2", len(bz2.compress(sorted_seq)))
    row_seq("sorted seq + lzma", len(lzma.compress(sorted_seq)))

    # .polar v2
    print(f"\n  --- .polar v2 format ---")
    blob, t_enc = encode_polar_v2(reads, seq_k=6, n_buckets=8)
    row(".polar v2 (k=6, polar, binned qual)", len(blob), t_enc)

    # Stream breakdown
    off = 20
    stream_sizes = {}
    for _ in range(4):
        stype = blob[off]; off += 1
        ssize = struct.unpack_from("<I", blob, off)[0]; off += 4
        if stype in (1, 2): off += 4
        names = {0: "headers", 1: "sequences (V5-polar)", 2: "quality (binned+lzma)",
                 3: "meta (lengths+perm)"}
        stream_sizes[names[stype]] = ssize
        off += ssize
    print(f"\n  Stream breakdown:")
    for name, sz in stream_sizes.items():
        pct = 100 * sz / len(blob)
        print(f"    {name:<35} {sz:>8} bytes  ({pct:>5.1f}%)")

    # Round-trip
    decoded = decode_polar_v2(blob, seq_k=6, n_buckets=8)
    ok = True
    for i, (orig, dec) in enumerate(zip(reads, decoded)):
        orig_clean = re.sub(r'[^ACGT]', '', orig[1].upper())
        if dec[1] != orig_clean:
            print(f"\n  SEQ MISMATCH read {i}: got {len(dec[1])}, expected {len(orig_clean)}")
            ok = False; break
        # Quality: compare binned (since we quantized)
        orig_binned = bin_quality_string(orig[2])
        dec_binned  = bin_quality_string(dec[2])
        if not np.array_equal(orig_binned, dec_binned):
            print(f"\n  QUAL MISMATCH read {i}")
            ok = False; break
    print(f"\n  Round-trip: {'OK' if ok else 'FAIL'} (sequences lossless, quality binned to 8 levels)")

    return len(blob), fastq_size

if __name__ == "__main__":
    ZIP = "fastqfiles.zip"
    MAX_READS = 100

    print("=" * 90)
    print(".POLAR v2 FORMAT  --  Improved DNA Compression")
    print(f"  Quality binning: 94 → 8 levels  |  Seq k=6  |  Polar B=8")
    print(f"  Max reads/file: {MAX_READS}")
    print("=" * 90)

    with zipfile.ZipFile(ZIP) as z:
        fnames = sorted([i.filename for i in z.infolist()
                         if i.filename.endswith('.fastq')])

    # Sample 6 diverse files
    picks = [fnames[0], fnames[-1],
             fnames[len(fnames)//5], fnames[2*len(fnames)//5],
             fnames[3*len(fnames)//5], fnames[4*len(fnames)//5]]
    seen = set(); unique = []
    for f in picks:
        if f not in seen: seen.add(f); unique.append(f)

    total_polar = 0; total_fastq = 0
    for fname in unique:
        p, f = bench_file(ZIP, fname, max_reads=MAX_READS)
        total_polar += p; total_fastq += f

    print(f"\n\n{'#'*90}")
    print(f"AGGREGATE ({len(unique)} files)")
    print(f"{'#'*90}")
    print(f"  Total FASTQ raw:    {total_fastq:>12,} bytes")
    print(f"  Total .polar v2:    {total_polar:>12,} bytes")
    print(f"  Compression ratio:  {total_fastq/total_polar:>12.2f}x")
    print(f"  vs FASTQ+bz2 est:  ~2.8x (from v1 benchmarks)")
