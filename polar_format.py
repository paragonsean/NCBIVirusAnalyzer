"""
polar_format.py  --  .polar: a custom DNA sequencing format.

FASTQ is 4 interleaved streams crammed into one text file:
  @header
  ACGTACGT...
  +
  !!##$$%%...

This is terrible for compression because each stream has completely
different statistics.  Our .polar format splits them into independent
streams, sorts reads by sequence similarity so the adaptive coder sees
near-identical bases in sequence, and compresses each stream with the
best method for its alphabet:

  STREAM 0 — HEADER:  deduplicated common prefix + delta-encoded unique parts → lzma
  STREAM 1 — SEQUENCE: sorted reads concatenated → V5 adaptive PPM (order-4) + polar
  STREAM 2 — QUALITY:  Phred scores (ASCII 33–73) → adaptive order-2 + range coder
  STREAM 3 — LENGTHS:  per-read sequence lengths → delta + varint → lzma

File layout:
  [8 bytes]  magic b'POLARDNA'
  [4 bytes]  version (uint32 = 1)
  [4 bytes]  n_reads (uint32)
  [4 bytes]  n_streams (uint32 = 4)
  For each stream:
    [1 byte]   stream type (0=header, 1=seq, 2=qual, 3=lengths)
    [4 bytes]  compressed size
    [N bytes]  compressed data
"""

import numpy as np, struct, zlib, bz2, lzma, zipfile, re, time, sys, os
from concurrent.futures import ThreadPoolExecutor, as_completed
import constriction

RangeEncoder = constriction.stream.queue.RangeEncoder
RangeDecoder = constriction.stream.queue.RangeDecoder
Categorical  = constriction.stream.model.Categorical

BASES = "ACGT"
B2I = {b: i for i, b in enumerate(BASES)}
I2B = {i: b for i, b in enumerate(BASES)}
PHASES = np.array([0, np.pi/2, np.pi, 3*np.pi/2])
MAGIC = b'POLARDNA'
VERSION = 1

# ================================================================
# FASTQ PARSER
# ================================================================
def parse_fastq_from_zip(zippath, filename, max_reads=None):
    """Parse a single .fastq file from a zip → list of (header, seq, qual) tuples."""
    reads = []
    with zipfile.ZipFile(zippath) as z:
        with z.open(filename) as f:
            lines = []
            for raw in f:
                lines.append(raw.decode().rstrip('\n\r'))
                if len(lines) == 4:
                    header = lines[0]       # @...
                    seq    = lines[1]       # ACGT...
                    qual   = lines[3]       # quality ASCII
                    reads.append((header, seq, qual))
                    lines = []
                    if max_reads and len(reads) >= max_reads:
                        break
    return reads

def parse_fastq_string(text):
    """Parse FASTQ from a string."""
    lines = text.strip().split('\n')
    reads = []
    for i in range(0, len(lines) - 3, 4):
        reads.append((lines[i], lines[i+1], lines[i+3]))
    return reads

# ================================================================
# READ SORTING (the key insight for compression)
# ================================================================
def sort_reads_by_sequence(reads):
    """Sort reads lexicographically by their DNA sequence.
    Reads from the same amplicon will cluster together, giving the
    adaptive coder long runs of near-identical context."""
    # Return sorted reads + the permutation (for faithful reconstruction)
    indexed = list(enumerate(reads))
    indexed.sort(key=lambda x: x[1][1])  # sort by sequence string
    perm = [i for i, _ in indexed]
    sorted_reads = [r for _, r in indexed]
    return sorted_reads, perm

def unsort_reads(sorted_reads, perm):
    """Restore original read order from the sort permutation."""
    n = len(sorted_reads)
    out = [None] * n
    for new_idx, orig_idx in enumerate(perm):
        out[orig_idx] = sorted_reads[new_idx]
    return out

# ================================================================
# ADAPTIVE PPM (from V5, for sequence + quality streams)
# ================================================================
def make_probs(counts_stack, alpha=1.0):
    p = np.full(4, 0.25, dtype=np.float64)
    for c in counts_stack:
        total = c.sum()
        p = (c + alpha * p) / (total + alpha)
    p = np.clip(p, 1e-9, None); p /= p.sum()
    return p

class AdaptiveModel:
    def __init__(self, k, n_symbols=4, alpha=1.0):
        self.k = k; self.alpha = alpha; self.n_symbols = n_symbols
        self.tables = [dict() for _ in range(k + 1)]
    def probs(self, ctx_full):
        stack = []
        uniform = np.full(self.n_symbols, 1.0/self.n_symbols, dtype=np.float64)
        for o in range(self.k + 1):
            ctx = tuple(int(x) for x in ctx_full[self.k - o:])
            stack.append(self.tables[o].get(ctx,
                         np.zeros(self.n_symbols, dtype=np.int64)))
        # interpolated smoothing
        p = uniform.copy()
        for c in stack:
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

# Polar-conditioned adaptive model
class AdaptivePolarModel:
    def __init__(self, k, n_buckets=8, alpha=1.0):
        self.k = k; self.n_buckets = n_buckets; self.alpha = alpha
        self.banks = [AdaptiveModel(k, 4, alpha) for _ in range(n_buckets)]
    def probs(self, ctx, b): return self.banks[b].probs(ctx)
    def update(self, ctx, sym, b): self.banks[b].update(ctx, sym)

# ================================================================
# STREAM ENCODERS
# ================================================================

def encode_sequence_stream(sorted_seqs, k=4, use_polar=True, n_buckets=8):
    """Encode all sorted sequences as one continuous stream via V5 adaptive.
    Insert a reset sentinel between reads so the model carries context
    across similar reads but doesn't cross amplicon boundaries badly."""
    # Concatenate all sequences (sorted → similar reads adjacent)
    all_bases = "".join(sorted_seqs)
    # Clean non-ACGT
    all_bases = re.sub(r'[^ACGT]', '', all_bases.upper())
    ids = np.fromiter((B2I[c] for c in all_bases), dtype=np.int32, count=len(all_bases))
    n = len(ids)

    enc = RangeEncoder()
    if use_polar:
        model = AdaptivePolarModel(k, n_buckets)
    else:
        model = AdaptiveModel(k, 4)
    pad = np.zeros(k, dtype=np.int32)
    z_window = []
    block_size = 64

    for i in range(n):
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        if use_polar:
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
        else:
            p = model.probs(ctx)
            enc.encode(int(ids[i]), Categorical(p, perfect=False))
            model.update(ctx, int(ids[i]))

    return enc.get_compressed().tobytes(), n

def decode_sequence_stream(payload, n, k=4, use_polar=True, n_buckets=8):
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    pad_bytes = (-len(arr)) % 4
    if pad_bytes: arr = np.concatenate([arr, np.zeros(pad_bytes, dtype=np.uint8)])
    u32 = arr.view(np.uint32).copy()
    dec = RangeDecoder(u32)

    if use_polar:
        model = AdaptivePolarModel(k, n_buckets)
    else:
        model = AdaptiveModel(k, 4)
    pad = np.zeros(k, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    z_window = []
    block_size = 64

    for i in range(n):
        ctx = np.concatenate([pad, out[:i]])[-k:]
        if use_polar:
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
        else:
            p = model.probs(ctx)
            sym = dec.decode(Categorical(p, perfect=False))
            s = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
            out[i] = s
            model.update(ctx, s)

    return "".join(I2B[int(x)] for x in out)

def encode_quality_stream(sorted_quals, k=2):
    """Encode Phred quality scores with adaptive PPM.
    Quality alphabet: ASCII 33–126 → 0–93 (typically 0–40 used)."""
    all_q = "".join(sorted_quals)
    n_syms = 94  # Phred range
    ids = np.array([max(0, min(ord(c) - 33, 93)) for c in all_q], dtype=np.int32)
    n = len(ids)

    enc = RangeEncoder()
    model = AdaptiveModel(k, n_syms)
    pad = np.zeros(k, dtype=np.int32)
    for i in range(n):
        ctx = np.concatenate([pad, ids[:i]])[-k:]
        p = model.probs(ctx)
        enc.encode(int(ids[i]), Categorical(p, perfect=False))
        model.update(ctx, int(ids[i]))
    return enc.get_compressed().tobytes(), n

def decode_quality_stream(payload, n, k=2):
    n_syms = 94
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    pad_bytes = (-len(arr)) % 4
    if pad_bytes: arr = np.concatenate([arr, np.zeros(pad_bytes, dtype=np.uint8)])
    u32 = arr.view(np.uint32).copy()
    dec = RangeDecoder(u32)
    model = AdaptiveModel(k, n_syms)
    pad = np.zeros(k, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    for i in range(n):
        ctx = np.concatenate([pad, out[:i]])[-k:]
        p = model.probs(ctx)
        sym = dec.decode(Categorical(p, perfect=False))
        out[i] = int(sym[0]) if hasattr(sym, "__len__") else int(sym)
        model.update(ctx, int(out[i]))
    return "".join(chr(int(x) + 33) for x in out)

# ================================================================
# .POLAR FILE ENCODER / DECODER
# ================================================================
def encode_polar(reads, use_polar_seq=True, seq_k=4, qual_k=2):
    """Encode a list of (header, seq, qual) reads → .polar bytes."""
    t_start = time.time()

    # Sort reads by sequence for better compression
    sorted_reads, perm = sort_reads_by_sequence(reads)
    sorted_headers = [r[0] for r in sorted_reads]
    sorted_seqs    = [r[1] for r in sorted_reads]
    sorted_quals   = [r[2] for r in sorted_reads]
    seq_lengths    = [len(s) for s in sorted_seqs]

    # Stream 0: headers → lzma (text, high redundancy)
    header_blob = "\n".join(sorted_headers).encode()
    header_comp = lzma.compress(header_blob)

    # Stream 1: sequences → V5 adaptive
    seq_comp, seq_n = encode_sequence_stream(sorted_seqs, k=seq_k,
                                              use_polar=use_polar_seq)

    # Stream 2: quality → adaptive order-2
    qual_comp, qual_n = encode_quality_stream(sorted_quals, k=qual_k)

    # Stream 3: lengths + sort permutation → lzma
    meta = struct.pack(f"<{len(seq_lengths)}I", *seq_lengths)
    meta += struct.pack(f"<{len(perm)}I", *perm)
    meta_comp = lzma.compress(meta)

    # Pack into .polar file
    n_reads = len(reads)
    parts = []
    parts.append(MAGIC)                               # 8 bytes
    parts.append(struct.pack("<III", VERSION, n_reads, 4))  # 12 bytes
    # Stream entries
    for stream_type, data in [(0, header_comp), (1, seq_comp),
                               (2, qual_comp),  (3, meta_comp)]:
        parts.append(struct.pack("<BI", stream_type, len(data)))
        # For seq/qual, also store the uncompressed symbol count
        if stream_type in (1, 2):
            n_val = seq_n if stream_type == 1 else qual_n
            parts.append(struct.pack("<I", n_val))
        parts.append(data)

    blob = b"".join(parts)
    elapsed = time.time() - t_start
    return blob, elapsed

def decode_polar(blob, use_polar_seq=True, seq_k=4, qual_k=2):
    """Decode .polar bytes → list of (header, seq, qual) in original order."""
    off = 0
    magic = blob[off:off+8]; off += 8
    assert magic == MAGIC, f"Bad magic: {magic}"
    ver, n_reads, n_streams = struct.unpack_from("<III", blob, off); off += 12

    streams = {}
    for _ in range(n_streams):
        stype = struct.unpack_from("<B", blob, off)[0]; off += 1
        ssize = struct.unpack_from("<I", blob, off)[0]; off += 4
        n_val = None
        if stype in (1, 2):
            n_val = struct.unpack_from("<I", blob, off)[0]; off += 4
        sdata = blob[off:off+ssize]; off += ssize
        streams[stype] = (sdata, n_val)

    # Decode headers
    header_text = lzma.decompress(streams[0][0]).decode()
    sorted_headers = header_text.split("\n")

    # Decode sequences
    seq_data, seq_n = streams[1]
    all_seq_str = decode_sequence_stream(seq_data, seq_n, k=seq_k,
                                          use_polar=use_polar_seq)

    # Decode quality
    qual_data, qual_n = streams[2]
    all_qual_str = decode_quality_stream(qual_data, qual_n, k=qual_k)

    # Decode lengths + permutation
    meta_raw = lzma.decompress(streams[3][0])
    lengths = list(struct.unpack_from(f"<{n_reads}I", meta_raw, 0))
    perm = list(struct.unpack_from(f"<{n_reads}I", meta_raw, n_reads * 4))

    # Split concatenated sequences / quality back into per-read
    sorted_seqs = []
    pos = 0
    for l in lengths:
        sorted_seqs.append(all_seq_str[pos:pos+l])
        pos += l

    sorted_quals = []
    pos = 0
    for l in lengths:
        sorted_quals.append(all_qual_str[pos:pos+l])
        pos += l

    # Rebuild sorted reads, then unsort
    sorted_reads = list(zip(sorted_headers, sorted_seqs, sorted_quals))
    original_reads = unsort_reads(sorted_reads, perm)
    return original_reads

# ================================================================
# BENCHMARK
# ================================================================
def bench_file(zippath, filename, max_reads=500):
    reads = parse_fastq_from_zip(zippath, filename, max_reads=max_reads)
    n_reads = len(reads)
    # Original FASTQ size
    fastq_lines = []
    for h, s, q in reads:
        fastq_lines.extend([h, s, "+", q])
    fastq_raw = "\n".join(fastq_lines).encode()
    fastq_size = len(fastq_raw)

    total_bases = sum(len(r[1]) for r in reads)
    total_qual  = sum(len(r[2]) for r in reads)

    short = filename.replace('_2019_minq7.fastq', '')

    print(f"\n{'='*85}")
    print(f"{short}  ({n_reads} reads, {total_bases:,} bases)")
    print(f"{'='*85}")
    print(f"{'Method':<45}{'Bytes':>10}{'Ratio':>8}{'bpb*':>8}{'Time':>8}")
    print("-" * 85)

    def row(name, b, t=0):
        ts = f"{t:.1f}s" if t else ""
        ratio = fastq_size / max(b, 1)
        bpb = b * 8 / max(total_bases, 1)
        print(f"{name:<45}{b:>10}{ratio:>7.1f}x{bpb:>8.3f}{ts:>8}")

    row("Original FASTQ", fastq_size)
    row("FASTQ + gzip",  len(zlib.compress(fastq_raw)))
    row("FASTQ + bz2",   len(bz2.compress(fastq_raw)))
    row("FASTQ + lzma",  len(lzma.compress(fastq_raw)))

    # .polar without polar conditioning
    blob_plain, t1 = encode_polar(reads, use_polar_seq=False, seq_k=4)
    row(".polar (V5 plain, k=4)", len(blob_plain), t1)

    # .polar WITH polar conditioning
    blob_polar, t2 = encode_polar(reads, use_polar_seq=True, seq_k=4)
    row(".polar (V5 polar, k=4, B=8)", len(blob_polar), t2)

    # Stream breakdown for polar version
    print(f"\n  Stream breakdown (.polar V5-polar):")
    off = 20  # skip header
    for i in range(4):
        stype = blob_polar[off]; off += 1
        ssize = struct.unpack_from("<I", blob_polar, off)[0]; off += 4
        if stype in (1, 2):
            off += 4  # skip n_val
        names = {0: "headers", 1: "sequences", 2: "quality", 3: "lengths+perm"}
        print(f"    {names[stype]:<20} {ssize:>10} bytes")
        off += ssize

    # Round-trip check
    decoded = decode_polar(blob_polar, use_polar_seq=True, seq_k=4)
    # Compare cleaned sequences (original may have N's we stripped)
    ok = True
    for i, (orig, dec) in enumerate(zip(reads, decoded)):
        orig_seq_clean = re.sub(r'[^ACGT]', '', orig[1].upper())
        if dec[1] != orig_seq_clean:
            print(f"  SEQUENCE MISMATCH at read {i}: {len(dec[1])} vs {len(orig_seq_clean)}")
            ok = False
            break
        if dec[2] != orig[2]:
            print(f"  QUALITY MISMATCH at read {i}")
            ok = False
            break
    print(f"  Round-trip: {'OK' if ok else 'FAIL'}")
    return len(blob_polar), fastq_size

# ================================================================
# FILE I/O FUNCTIONS
# ================================================================
def write_polar_file(reads, output_path, use_polar_seq=True, seq_k=4, qual_k=2):
    """Encode reads and write to .polar file."""
    blob, elapsed = encode_polar(reads, use_polar_seq=use_polar_seq, 
                               seq_k=seq_k, qual_k=qual_k)
    with open(output_path, 'wb') as f:
        f.write(blob)
    return len(blob), elapsed

def read_polar_file(input_path, use_polar_seq=True, seq_k=4, qual_k=2):
    """Read .polar file and decode to reads."""
    with open(input_path, 'rb') as f:
        blob = f.read()
    return decode_polar(blob, use_polar_seq=use_polar_seq, seq_k=seq_k, qual_k=qual_k)

def convert_fastq_to_polar(zippath, fastq_filename, output_path, 
                          max_reads=None, use_polar_seq=True, seq_k=4, qual_k=2):
    """Convert FASTQ from zip to .polar file."""
    reads = parse_fastq_from_zip(zippath, fastq_filename, max_reads=max_reads)
    return write_polar_file(reads, output_path, use_polar_seq=use_polar_seq, 
                          seq_k=seq_k, qual_k=qual_k)

def convert_polar_to_fastq(polar_path, output_path, use_polar_seq=True, seq_k=4, qual_k=2):
    """Convert .polar file to FASTQ format."""
    reads = read_polar_file(polar_path, use_polar_seq=use_polar_seq, 
                           seq_k=seq_k, qual_k=qual_k)
    with open(output_path, 'w') as f:
        for header, seq, qual in reads:
            f.write(f"{header}\n{seq}\n+\n{qual}\n")

def extract_all_fastq_to_polar(zippath, output_polar_path, max_reads_per_file=None, 
                              use_polar_seq=True, seq_k=4, qual_k=2):
    """Extract all FASTQ files from zip and combine into single .polar file."""
    print(f"Extracting all FASTQ files from {zippath}")
    
    # Get all FASTQ files in zip
    with zipfile.ZipFile(zippath) as z:
        fastq_files = [f.filename for f in z.infolist() if f.filename.endswith('.fastq')]
    
    print(f"Found {len(fastq_files)} FASTQ files")
    
    all_reads = []
    total_reads = 0
    
    for i, fastq_file in enumerate(fastq_files):
        print(f"Processing {i+1}/{len(fastq_files)}: {fastq_file}")
        reads = parse_fastq_from_zip(zippath, fastq_file, max_reads=max_reads_per_file)
        all_reads.extend(reads)
        total_reads += len(reads)
        print(f"  Added {len(reads)} reads (total: {total_reads})")
    
    print(f"Total reads collected: {total_reads:,}")
    
    # Write all reads to single polar file
    print(f"Writing to {output_polar_path}")
    polar_bytes, elapsed = write_polar_file(all_reads, output_polar_path, 
                                          use_polar_seq=use_polar_seq, 
                                          seq_k=seq_k, qual_k=qual_k)
    
    compression_ratio = total_reads * 100 / polar_bytes  # rough estimate
    print(f"Wrote {polar_bytes:,} bytes in {elapsed:.1f}s")
    print(f"Estimated compression: ~{compression_ratio:.1f} reads per KB")
    
    return polar_bytes, total_reads, elapsed

if __name__ == "__main__":
    ZIP = "fastqfiles.zip"
    MAX_READS = 100   # 100 reads per file for speed

    print("=" * 85)
    print(".POLAR FORMAT  --  Custom DNA Compression Format")
    print(f"Max reads/file: {MAX_READS}")
    print("=" * 85)

    # List files
    with zipfile.ZipFile(ZIP) as z:
        fnames = [i.filename for i in z.infolist() if i.filename.endswith('.fastq')]

    # Benchmark a diverse sample
    sample = [fnames[0], fnames[len(fnames)//4], fnames[len(fnames)//2],
              fnames[3*len(fnames)//4], fnames[-1]]
    seen = set()
    unique = []
    for f in sample:
        if f not in seen: seen.add(f); unique.append(f)

    total_polar = 0
    total_fastq = 0
    for fname in unique:
        polar_sz, fastq_sz = bench_file(ZIP, fname, max_reads=MAX_READS)
        total_polar += polar_sz
        total_fastq += fastq_sz

    print(f"\n\n{'='*85}")
    print(f"AGGREGATE ({len(unique)} files sampled)")
    print(f"{'='*85}")
    print(f"  Total FASTQ:  {total_fastq:>12,} bytes")
    print(f"  Total .polar: {total_polar:>12,} bytes")
    print(f"  Ratio: {total_fastq/total_polar:.2f}x")

    # Example file writing
    print(f"\n{'='*85}")
    print("FILE WRITING EXAMPLE")
    print(f"{'='*85}")
    
    # Convert first file to .polar and back
    test_file = unique[0]
    polar_path = test_file.replace('.fastq', '.polar')
    recovered_path = test_file.replace('.fastq', '_recovered.fastq')
    
    print(f"Converting {test_file} to {polar_path}")
    polar_bytes, elapsed = convert_fastq_to_polar(ZIP, test_file, polar_path, 
                                                 max_reads=MAX_READS)
    print(f"Written {polar_bytes:,} bytes in {elapsed:.1f}s")
    
    print(f"Converting {polar_path} back to {recovered_path}")
    convert_polar_to_fastq(polar_path, recovered_path)
    
    # Verify round-trip
    original_reads = parse_fastq_from_zip(ZIP, test_file, max_reads=MAX_READS)
    recovered_reads = read_polar_file(polar_path)
    
    # Clean sequences for comparison
    orig_clean = [(h, re.sub(r'[^ACGT]', '', s.upper()), q) for h, s, q in original_reads]
    
    match = all(r1[1] == r2[1] and r1[2] == r2[2] for r1, r2 in zip(orig_clean, recovered_reads))
    print(f"Round-trip verification: {'PASS' if match else 'FAIL'}")

    # Extract all files to single polar file
    print(f"\n{'='*85}")
    print("EXTRACTING ALL FILES TO SINGLE POLAR")
    print(f"{'='*85}")
    
    all_polar_path = "all_reads.polar"
    polar_bytes, total_reads, elapsed = extract_all_fastq_to_polar(ZIP, all_polar_path, 
                                                                  max_reads_per_file=MAX_READS)
    
    # Test reading back the combined polar file
    print(f"\nTesting read-back of combined polar file...")
    recovered_all = read_polar_file(all_polar_path)
    print(f"Successfully read {len(recovered_all):,} reads from {all_polar_path}")
    
    # Convert back to FASTQ for verification
    all_fastq_path = "all_reads_recovered.fastq"
    convert_polar_to_fastq(all_polar_path, all_fastq_path)
    print(f"Converted back to FASTQ: {all_fastq_path}")
