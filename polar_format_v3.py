"""
polar_format_v2_magnitude.py  --  .polar v2 with Confidence-Weighted Averaging.

Stream layout:
  STREAM 0 — HEADERS:   lzma-compressed text
  STREAM 1 — SEQUENCES: V5 adaptive PPM (k=6) with MAGNITUDE POLAR conditioning
  STREAM 2 — QUALITY:   binned (8 levels) → lzma
  STREAM 3 — META:      read lengths + sort permutation + quality bin table → lzma
"""

import numpy as np, struct, zlib, bz2, lzma, zipfile, re, time
import constriction
from tqdm import tqdm

RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical  = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
I2B = {i: b for i, b in enumerate(BASES)}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
MAGIC = b'POLAR2MG' # Updated Magic Bytes for Magnitude version
VERSION = 2

# ================================================================
# QUALITY BINNING
# ================================================================
QUAL_BIN_EDGES = [0, 6, 10, 15, 20, 25, 30, 37, 94]
QUAL_BIN_REPS  = [2, 7, 12, 17, 22, 27, 33, 40]

def phred_to_bin(q):
    for i in range(len(QUAL_BIN_EDGES) - 1):
        if q < QUAL_BIN_EDGES[i + 1]:
            return i
    return 7

_PHRED_TO_BIN = np.array([phred_to_bin(q) for q in range(94)], dtype=np.uint8)

def bin_quality_string(qstr):
    return np.array([_PHRED_TO_BIN[max(0, min(ord(c) - 33, 93))] for c in qstr], dtype=np.uint8)

def unbin_quality_string(bins):
    return "".join(chr(QUAL_BIN_REPS[int(b)] + 33) for b in bins)

# ================================================================
# FASTQ PARSER & SORTING
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
# ADAPTIVE PPM + MAGNITUDE POLAR 
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

def encode_seq_v2(sorted_seqs, all_binned_quals, k=6, n_buckets=8):
    all_bases = re.sub(r'[^ACGT]', '', "".join(sorted_seqs).upper())
    ids = np.fromiter((B2I[c] for c in all_bases), dtype=np.int32, count=len(all_bases))
    n = len(ids)

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
        
        # --- THE MAGNITUDE UPGRADE ---
        q_bin = all_binned_quals[i] if i < len(all_binned_quals) else 7
        magnitude = (q_bin + 1) / 8.0 
        z_window.append(magnitude * np.exp(1j * PHASES[ids[i]]))
        
        if len(z_window) > block_size: z_window.pop(0)

    return enc.get_compressed().tobytes(), n

def decode_seq_v2(payload, n, all_binned_quals, k=6, n_buckets=8):
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
        
        # --- THE MAGNITUDE UPGRADE ---
        q_bin = all_binned_quals[i] if i < len(all_binned_quals) else 7
        magnitude = (q_bin + 1) / 8.0 
        z_window.append(magnitude * np.exp(1j * PHASES[s]))
        
        if len(z_window) > block_size: z_window.pop(0)

    return "".join(I2B[int(x)] for x in out)

# ================================================================
# .POLAR V2 ENCODER / DECODER
# ================================================================
def encode_polar_v2(reads, seq_k=6, n_buckets=8):
    t0 = time.time()
    sorted_reads, perm = sort_reads(reads)
    sorted_headers = [r[0] for r in sorted_reads]
    sorted_seqs    = [r[1] for r in sorted_reads]
    sorted_quals   = [r[2] for r in sorted_reads]

    cleaned_seqs = [re.sub(r'[^ACGT]', '', s.upper()) for s in sorted_seqs]
    seq_lengths  = [len(s) for s in cleaned_seqs]

    # STREAM 0: headers
    header_comp = lzma.compress(("\n".join(sorted_headers)).encode())

    # STREAM 2: quality MUST BE ENCODED FIRST so we can pass it to Stream 1
    all_binned = np.concatenate([bin_quality_string(q) for q in sorted_quals])
    qual_comp = lzma.compress(bytes(all_binned))
    qual_n = len(all_binned)

    # STREAM 1: sequences (Pass all_binned in for Magnitude logic)
    seq_comp, seq_n = encode_seq_v2(cleaned_seqs, all_binned, k=seq_k, n_buckets=n_buckets)

    # STREAM 3: meta
    qual_lengths = [len(q) for q in sorted_quals]
    meta = struct.pack(f"<{len(seq_lengths)}I", *seq_lengths)
    meta += struct.pack(f"<{len(qual_lengths)}I", *qual_lengths)
    meta += struct.pack(f"<{len(perm)}I", *perm)
    meta_comp = lzma.compress(meta)

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

    return b"".join(parts), time.time() - t0

def decode_polar_v2(blob, seq_k=6, n_buckets=8):
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

    # STREAM 0: Headers
    sorted_headers = lzma.decompress(streams[0][0]).decode().split("\n")

    # STREAM 2: Quality MUST DECODE FIRST
    all_binned = np.frombuffer(lzma.decompress(streams[2][0]), dtype=np.uint8)

    # STREAM 1: Sequences (Pass all_binned in for Magnitude logic)
    all_seq = decode_seq_v2(streams[1][0], streams[1][1], all_binned, k=seq_k, n_buckets=n_buckets)

    # STREAM 3: Meta
    meta_raw = lzma.decompress(streams[3][0])
    seq_lengths  = list(struct.unpack_from(f"<{n_reads}I", meta_raw, 0))
    qual_lengths = list(struct.unpack_from(f"<{n_reads}I", meta_raw, n_reads*4))
    perm         = list(struct.unpack_from(f"<{n_reads}I", meta_raw, n_reads*8))

    sorted_seqs = []
    pos = 0
    for l in seq_lengths:
        sorted_seqs.append(all_seq[pos:pos+l]); pos += l

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

    fastq_lines = []
    for h, s, q in reads:
        fastq_lines.extend([h, s, "+", q])
    fastq_raw = "\n".join(fastq_lines).encode()
    fastq_size = len(fastq_raw)

    short = filename.replace('_2019_minq7.fastq', '')

    print(f"\n{'='*90}")
    print(f"{short}  ({n_reads} reads)")
    print(f"{'='*90}")
    print(f"{'Method':<50}{'Bytes':>10}{'Ratio':>8}{'Time':>8}")
    print("-" * 90)

    def row(name, b, t=0):
        ts = f"{t:.1f}s" if t else ""
        ratio = fastq_size / max(b, 1)
        print(f"{name:<50}{b:>10}{ratio:>7.1f}x{ts:>8}")

    row("Original FASTQ", fastq_size)
    row("FASTQ + lzma", len(lzma.compress(fastq_raw)))

    blob, t_enc = encode_polar_v2(reads, seq_k=6, n_buckets=8)
    row(".polar v2 (k=6, magnitude polar)", len(blob), t_enc)

    decoded = decode_polar_v2(blob, seq_k=6, n_buckets=8)
    ok = True
    for i, (orig, dec) in enumerate(zip(reads, decoded)):
        orig_clean = re.sub(r'[^ACGT]', '', orig[1].upper())
        if dec[1] != orig_clean:
            ok = False; break
        orig_binned = bin_quality_string(orig[2])
        dec_binned  = bin_quality_string(dec[2])
        if not np.array_equal(orig_binned, dec_binned):
            ok = False; break
            
    print(f"\n  Round-trip: {'OK' if ok else 'FAIL'}")
    return len(blob), fastq_size

if __name__ == "__main__":
    ZIP = "fastqfiles.zip"
    MAX_READS = 100
    with zipfile.ZipFile(ZIP) as z:
        fnames = [i.filename for i in z.infolist() if i.filename.endswith('.fastq')]
    bench_file(ZIP, fnames[0], max_reads=MAX_READS)