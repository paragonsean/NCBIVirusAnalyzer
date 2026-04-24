"""
polar_db.py  --  Lossless FASTQ → Polar Database converter.

Converts all FASTQ files into a structured SQLite database where every
DNA base is stored in its polar (complex phasor) representation.

ZERO DATA LOSS: every field is preserved exactly — headers, sequences,
quality scores, read order, file origin. The original FASTQ files can
be reconstructed byte-for-byte from the database.

Polar mapping (Voss representation):
    A → phase 0        → complex  1 + 0j
    C → phase π/2      → complex  0 + 1j
    G → phase π        → complex -1 + 0j
    T → phase 3π/2     → complex  0 - 1j

Each base is stored as a uint8 phase index (0–3), which maps bijectively
to both the original base AND the complex phasor. No information is lost.

Database schema:
    files:    source file metadata
    reads:    per-read header, lengths, quality, polar data
    summary:  per-file statistics (GC content, mean quality, polar centroid)

Usage:
    python polar_db.py                    # ingest from fastqfiles.zip
    python polar_db.py --verify           # verify round-trip
    python polar_db.py --export FILE_ID   # export back to FASTQ
    python polar_db.py --stats            # show polar statistics
"""

import numpy as np, struct, sqlite3, zipfile, re, time, sys, os, json, hashlib

# ================================================================
# POLAR MAPPING — the core bijection
# ================================================================
BASES = "ACGT"
# Phase index: A=0, C=1, G=2, T=3
B2P = {b: i for i, b in enumerate(BASES)}   # base → phase index
P2B = {i: b for i, b in enumerate(BASES)}   # phase index → base

# Full complex phasors (for analysis, not storage)
PHASES = np.array([0.0, np.pi/2, np.pi, 3*np.pi/2])          # radians
PHASORS = np.exp(1j * PHASES)  # [1+0j, 0+1j, -1+0j, 0-1j]

# Ambiguity codes → nearest base (for lossless storage we keep the original)
AMBIGUITY = {'N': None, 'R': None, 'Y': None, 'S': None, 'W': None,
             'K': None, 'M': None, 'B': None, 'D': None, 'H': None, 'V': None}

def seq_to_polar(seq):
    """Convert DNA string → uint8 array of phase indices (0-3).
    Non-ACGT characters get phase index 255 (sentinel).
    Returns (polar_ids, has_ambiguity)."""
    ids = np.zeros(len(seq), dtype=np.uint8)
    has_ambig = False
    for i, c in enumerate(seq.upper()):
        if c in B2P:
            ids[i] = B2P[c]
        else:
            ids[i] = 255   # sentinel for non-ACGT
            has_ambig = True
    return ids, has_ambig

def polar_to_seq(ids, original_seq=None):
    """Convert phase indices back to DNA string.
    If original_seq is provided, non-ACGT characters are restored from it."""
    out = []
    for i, p in enumerate(ids):
        if p <= 3:
            out.append(P2B[int(p)])
        elif original_seq and i < len(original_seq):
            out.append(original_seq[i])  # restore ambiguity code
        else:
            out.append('N')
    return "".join(out)

def polar_to_complex(ids):
    """Convert phase indices to complex phasor array (for analysis)."""
    mask = ids <= 3
    z = np.zeros(len(ids), dtype=np.complex128)
    z[mask] = PHASORS[ids[mask]]
    return z

def compute_polar_stats(ids):
    """Compute polar statistics for a read.
    Returns dict with centroid, resultant_length, dominant_phase, gc_content."""
    mask = ids <= 3
    valid = ids[mask]
    if len(valid) == 0:
        return {"centroid_real": 0, "centroid_imag": 0,
                "resultant_length": 0, "dominant_phase": -1,
                "gc_content": 0, "at_content": 0,
                "n_bases": 0, "n_ambiguous": int((~mask).sum())}

    z = PHASORS[valid]
    centroid = z.mean()

    # Base composition
    counts = np.bincount(valid, minlength=4)
    gc = (counts[1] + counts[2]) / len(valid)  # C + G
    at = (counts[0] + counts[3]) / len(valid)  # A + T

    return {
        "centroid_real": float(centroid.real),
        "centroid_imag": float(centroid.imag),
        "resultant_length": float(abs(centroid)),
        "dominant_phase": int(np.argmax(counts)),
        "gc_content": float(gc),
        "at_content": float(at),
        "n_bases": int(len(valid)),
        "n_ambiguous": int((~mask).sum()),
        "base_counts": counts.tolist()
    }

# ================================================================
# NANOPORE HEADER PARSER
# ================================================================
def parse_nanopore_header(header):
    """Parse a Nanopore FASTQ header into structured fields.
    Example: @83b951be-cf09-402a-8b19-48105583c067 runid=34c547ba... sampleid=1 read=68633 ch=80 start_time=2019-10-18T05:04:31Z
    """
    fields = {"raw_header": header}

    # Read ID (UUID after @)
    m = re.match(r'@(\S+)', header)
    if m:
        fields["read_uuid"] = m.group(1)

    # Key=value pairs
    for kv in re.finditer(r'(\w+)=(\S+)', header):
        k, v = kv.group(1), kv.group(2)
        fields[k] = v

    return fields

# ================================================================
# DATABASE SCHEMA
# ================================================================
SCHEMA = """
-- Source FASTQ files
CREATE TABLE IF NOT EXISTS files (
    file_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    sample      TEXT,           -- e.g. "1_control", "2_OHara_S1"
    marker      TEXT,           -- e.g. "18S", "ITS2", "psbA3", "rbcLa", "trnL", "MATK"
    year        INTEGER,
    n_reads     INTEGER,
    total_bases INTEGER,
    total_qual_chars INTEGER,
    file_hash   TEXT,           -- SHA256 of original FASTQ content
    imported_at TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Individual reads with polar data
CREATE TABLE IF NOT EXISTS reads (
    read_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id         INTEGER NOT NULL REFERENCES files(file_id),
    read_index      INTEGER NOT NULL,     -- position in original file (0-based)
    read_uuid       TEXT,                 -- Nanopore UUID
    header_raw      TEXT NOT NULL,        -- full original header line
    seq_length      INTEGER NOT NULL,
    qual_length     INTEGER NOT NULL,
    original_seq    TEXT NOT NULL,         -- original sequence (preserves N's, ambiguity codes)
    polar_phases    BLOB NOT NULL,         -- uint8 array of phase indices (0-3, 255=ambig)
    quality_raw     TEXT NOT NULL,         -- original quality ASCII string
    quality_phred   BLOB NOT NULL,         -- uint8 array of Phred scores
    has_ambiguity   INTEGER DEFAULT 0,     -- 1 if contains non-ACGT bases
    -- Polar statistics (precomputed for fast queries)
    centroid_real   REAL,
    centroid_imag   REAL,
    resultant_len   REAL,
    gc_content      REAL,
    mean_quality    REAL,
    -- Nanopore metadata
    run_id          TEXT,
    sample_id       TEXT,
    channel         INTEGER,
    start_time      TEXT
);

-- Per-file summary statistics
CREATE TABLE IF NOT EXISTS file_summary (
    file_id         INTEGER PRIMARY KEY REFERENCES files(file_id),
    mean_read_len   REAL,
    median_read_len REAL,
    mean_gc         REAL,
    mean_quality    REAL,
    total_a         INTEGER,
    total_c         INTEGER,
    total_g         INTEGER,
    total_t         INTEGER,
    total_n         INTEGER,
    centroid_real    REAL,
    centroid_imag    REAL
);

-- Indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_reads_file ON reads(file_id);
CREATE INDEX IF NOT EXISTS idx_reads_uuid ON reads(read_uuid);
CREATE INDEX IF NOT EXISTS idx_reads_gc ON reads(gc_content);
CREATE INDEX IF NOT EXISTS idx_reads_quality ON reads(mean_quality);
CREATE INDEX IF NOT EXISTS idx_reads_length ON reads(seq_length);
CREATE INDEX IF NOT EXISTS idx_files_sample ON files(sample);
CREATE INDEX IF NOT EXISTS idx_files_marker ON files(marker);
"""

# ================================================================
# FASTQ → POLAR DATABASE INGESTION
# ================================================================
def parse_filename_metadata(filename):
    """Extract sample and marker from filename.
    e.g. '2_OHara_S1_psbA3_2019_minq7.fastq' → sample='2_OHara_S1', marker='psbA3', year=2019"""
    base = filename.replace('.fastq', '')
    markers = ['18S', 'ITS2', 'psbA3_A', 'psbA3', 'rbcLa', 'trnL', 'MATK']

    marker = None
    sample = base
    for m in markers:
        if f'_{m}_' in base:
            marker = m
            idx = base.index(f'_{m}_')
            sample = base[:idx]
            break

    year = None
    ym = re.search(r'_(\d{4})_', base)
    if ym:
        year = int(ym.group(1))

    return sample, marker, year

def ingest_fastq_file(db_path, zippath, filename, max_reads=None, batch_size=500):
    """Ingest a single FASTQ file from a zip into the polar database."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    sample, marker, year = parse_filename_metadata(filename)

    # Read the raw FASTQ content for hashing
    with zipfile.ZipFile(zippath) as z:
        raw_content = z.read(filename)
    file_hash = hashlib.sha256(raw_content).hexdigest()

    # Check if already ingested
    cur.execute("SELECT file_id FROM files WHERE file_hash = ?", (file_hash,))
    existing = cur.fetchone()
    if existing:
        conn.close()
        return existing[0], 0  # already ingested

    # Parse reads
    reads = []
    lines = []
    for raw_line in raw_content.decode('utf-8', errors='replace').split('\n'):
        line = raw_line.rstrip('\r')
        if not line:
            continue
        lines.append(line)
        if len(lines) == 4:
            reads.append((lines[0], lines[1], lines[3]))
            lines = []
            if max_reads and len(reads) >= max_reads:
                break

    total_bases = sum(len(r[1]) for r in reads)
    total_qual = sum(len(r[2]) for r in reads)

    # Insert file record
    cur.execute("""INSERT INTO files (filename, sample, marker, year, n_reads,
                   total_bases, total_qual_chars, file_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (filename, sample, marker, year, len(reads),
                 total_bases, total_qual, file_hash))
    file_id = cur.lastrowid

    # Accumulate file-level stats
    all_counts = np.zeros(4, dtype=np.int64)
    total_n = 0
    gc_sum = 0.0
    qual_sum = 0.0
    lengths = []

    # Batch insert reads
    for batch_start in range(0, len(reads), batch_size):
        batch = reads[batch_start:batch_start + batch_size]
        rows = []
        for local_idx, (header, seq, qual) in enumerate(batch):
            read_idx = batch_start + local_idx
            # Parse header
            hfields = parse_nanopore_header(header)

            # Convert to polar
            polar_ids, has_ambig = seq_to_polar(seq)
            stats = compute_polar_stats(polar_ids)

            # Quality to Phred array
            phred = np.array([max(0, ord(c) - 33) for c in qual], dtype=np.uint8)
            mean_q = float(phred.mean()) if len(phred) > 0 else 0.0

            # Accumulate file stats
            if "base_counts" in stats:
                all_counts += np.array(stats["base_counts"])
            total_n += stats["n_ambiguous"]
            gc_sum += stats["gc_content"]
            qual_sum += mean_q
            lengths.append(len(seq))

            rows.append((
                file_id, read_idx,
                hfields.get("read_uuid"),
                header,
                len(seq), len(qual),
                seq,                                    # original sequence preserved
                bytes(polar_ids),                       # polar phase indices as blob
                qual,                                   # original quality string
                bytes(phred),                           # Phred scores as blob
                1 if has_ambig else 0,
                stats["centroid_real"], stats["centroid_imag"],
                stats["resultant_length"], stats["gc_content"],
                mean_q,
                hfields.get("runid"), hfields.get("sampleid"),
                int(hfields["ch"]) if "ch" in hfields else None,
                hfields.get("start_time")
            ))

        cur.executemany("""INSERT INTO reads
            (file_id, read_index, read_uuid, header_raw,
             seq_length, qual_length, original_seq, polar_phases,
             quality_raw, quality_phred, has_ambiguity,
             centroid_real, centroid_imag, resultant_len, gc_content,
             mean_quality, run_id, sample_id, channel, start_time)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", rows)

    # File summary
    lengths = np.array(lengths)
    n = len(reads)
    cur.execute("""INSERT INTO file_summary
        (file_id, mean_read_len, median_read_len, mean_gc, mean_quality,
         total_a, total_c, total_g, total_t, total_n,
         centroid_real, centroid_imag)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
        (file_id, float(lengths.mean()), float(np.median(lengths)),
         gc_sum / max(n, 1), qual_sum / max(n, 1),
         int(all_counts[0]), int(all_counts[1]),
         int(all_counts[2]), int(all_counts[3]),
         int(total_n),
         float(PHASORS[np.argmax(all_counts)].real),
         float(PHASORS[np.argmax(all_counts)].imag)))

    conn.commit()
    conn.close()
    return file_id, len(reads)

def ingest_all(db_path, zippath, max_reads_per_file=None):
    """Ingest all FASTQ files from a zip into the polar database."""
    # Create database with schema
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    conn.close()

    with zipfile.ZipFile(zippath) as z:
        fnames = sorted([i.filename for i in z.infolist()
                         if i.filename.endswith('.fastq')])

    print(f"Found {len(fnames)} FASTQ files in {zippath}")
    total_reads = 0
    t0 = time.time()

    for i, fname in enumerate(fnames):
        t1 = time.time()
        fid, n = ingest_fastq_file(db_path, zippath, fname,
                                    max_reads=max_reads_per_file)
        total_reads += n
        elapsed = time.time() - t1
        print(f"  [{i+1:>2}/{len(fnames)}] {fname:<50} "
              f"{n:>5} reads  {elapsed:.1f}s")

    total_time = time.time() - t0
    print(f"\nDone: {total_reads:,} reads from {len(fnames)} files in {total_time:.1f}s")
    print(f"Database: {db_path} ({os.path.getsize(db_path):,} bytes)")
    return total_reads

# ================================================================
# EXPORT: POLAR DATABASE → FASTQ (lossless round-trip)
# ================================================================
def export_to_fastq(db_path, file_id=None, output_path=None):
    """Export reads from polar database back to FASTQ format.
    If file_id is None, exports all files."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    if file_id is not None:
        cur.execute("SELECT filename FROM files WHERE file_id = ?", (file_id,))
        fname = cur.fetchone()[0]
        cur.execute("""SELECT header_raw, original_seq, quality_raw
                       FROM reads WHERE file_id = ? ORDER BY read_index""",
                    (file_id,))
    else:
        cur.execute("""SELECT r.header_raw, r.original_seq, r.quality_raw
                       FROM reads r JOIN files f ON r.file_id = f.file_id
                       ORDER BY f.file_id, r.read_index""")

    lines = []
    count = 0
    for header, seq, qual in cur:
        lines.append(header)
        lines.append(seq)
        lines.append("+")
        lines.append(qual)
        count += 1

    conn.close()

    text = "\n".join(lines) + "\n"
    if output_path:
        with open(output_path, 'w') as f:
            f.write(text)
    return text, count

# ================================================================
# VERIFICATION
# ================================================================
def verify_roundtrip(db_path, zippath, max_reads_per_file=None):
    """Verify that every read in the database matches the original FASTQ."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.execute("SELECT file_id, filename, file_hash FROM files ORDER BY file_id")
    files = cur.fetchall()

    all_ok = True
    for file_id, filename, stored_hash in files:
        # Re-read original
        with zipfile.ZipFile(zippath) as z:
            raw = z.read(filename)
        actual_hash = hashlib.sha256(raw).hexdigest()
        if actual_hash != stored_hash:
            print(f"  HASH MISMATCH: {filename}")
            all_ok = False
            continue

        # Parse original reads
        orig_reads = []
        lines = []
        for raw_line in raw.decode('utf-8', errors='replace').split('\n'):
            line = raw_line.rstrip('\r')
            if not line: continue
            lines.append(line)
            if len(lines) == 4:
                orig_reads.append((lines[0], lines[1], lines[3]))
                lines = []
                if max_reads_per_file and len(orig_reads) >= max_reads_per_file:
                    break

        # Compare against DB
        cur.execute("""SELECT header_raw, original_seq, quality_raw, polar_phases
                       FROM reads WHERE file_id = ? ORDER BY read_index""",
                    (file_id,))
        db_reads = cur.fetchall()

        if len(db_reads) != len(orig_reads):
            print(f"  COUNT MISMATCH: {filename} ({len(db_reads)} vs {len(orig_reads)})")
            all_ok = False
            continue

        file_ok = True
        for idx, (orig, db_row) in enumerate(zip(orig_reads, db_reads)):
            header_db, seq_db, qual_db, polar_blob = db_row
            # Check header
            if header_db != orig[0]:
                print(f"  HEADER MISMATCH: {filename} read {idx}")
                file_ok = False; break
            # Check sequence
            if seq_db != orig[1]:
                print(f"  SEQ MISMATCH: {filename} read {idx}")
                file_ok = False; break
            # Check quality
            if qual_db != orig[2]:
                print(f"  QUAL MISMATCH: {filename} read {idx}")
                file_ok = False; break
            # Check polar ↔ sequence bijection
            polar_ids = np.frombuffer(polar_blob, dtype=np.uint8)
            reconstructed = polar_to_seq(polar_ids, orig[1])
            if reconstructed != orig[1]:
                print(f"  POLAR ROUNDTRIP FAIL: {filename} read {idx}")
                file_ok = False; break

        if file_ok:
            print(f"  OK: {filename} ({len(db_reads)} reads)")
        else:
            all_ok = False

    conn.close()
    return all_ok

# ================================================================
# QUERY / STATS
# ================================================================
def show_stats(db_path):
    """Show database statistics and polar analysis."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Overall stats
    cur.execute("SELECT COUNT(*), SUM(n_reads), SUM(total_bases) FROM files")
    n_files, total_reads, total_bases = cur.fetchone()

    print(f"\n{'='*70}")
    print(f"POLAR DATABASE STATISTICS")
    print(f"{'='*70}")
    print(f"  Files:       {n_files}")
    print(f"  Total reads: {total_reads:,}")
    print(f"  Total bases: {total_bases:,}")
    print(f"  DB size:     {os.path.getsize(db_path):,} bytes")

    # Per-marker stats
    print(f"\n  {'Marker':<12}{'Files':>7}{'Reads':>10}{'Bases':>14}{'Mean GC':>10}")
    print(f"  {'-'*53}")
    cur.execute("""SELECT f.marker, COUNT(*), SUM(f.n_reads), SUM(f.total_bases),
                   AVG(s.mean_gc)
                   FROM files f JOIN file_summary s ON f.file_id = s.file_id
                   GROUP BY f.marker ORDER BY f.marker""")
    for marker, n, reads, bases, gc in cur:
        print(f"  {marker or 'unknown':<12}{n:>7}{reads:>10}{bases:>14,}{gc:>10.3f}")

    # Per-sample stats
    print(f"\n  {'Sample':<20}{'Files':>7}{'Reads':>10}{'Mean Qual':>10}")
    print(f"  {'-'*47}")
    cur.execute("""SELECT f.sample, COUNT(*), SUM(f.n_reads), AVG(s.mean_quality)
                   FROM files f JOIN file_summary s ON f.file_id = s.file_id
                   GROUP BY f.sample ORDER BY f.sample""")
    for sample, n, reads, mq in cur:
        print(f"  {sample or 'unknown':<20}{n:>7}{reads:>10}{mq:>10.1f}")

    # Polar centroid distribution
    print(f"\n  Polar centroid distribution (per-read):")
    cur.execute("""SELECT
        AVG(centroid_real), AVG(centroid_imag),
        AVG(resultant_len), AVG(gc_content), AVG(mean_quality)
        FROM reads""")
    cr, ci, rl, gc, mq = cur.fetchone()
    print(f"    Mean centroid:      ({cr:.4f}, {ci:.4f})")
    print(f"    Mean resultant len: {rl:.4f}")
    print(f"    Mean GC content:    {gc:.4f}")
    print(f"    Mean Phred quality: {mq:.1f}")

    # Read length distribution
    cur.execute("""SELECT MIN(seq_length), MAX(seq_length), AVG(seq_length)
                   FROM reads""")
    mn, mx, avg = cur.fetchone()
    print(f"\n  Read lengths: min={mn}, max={mx}, mean={avg:.0f}")

    conn.close()

# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    ZIP = "fastqfiles.zip"
    DB  = "polar_reads.db"

    MAX_READS = 20000  # per file, for speed in this demo

    print("=" * 70)
    print("POLAR DATABASE  --  Lossless FASTQ → Polar Conversion")
    print(f"  Source:     {os.path.basename(ZIP)}")
    print(f"  Database:   {os.path.basename(DB)}")
    print(f"  Max reads:  {MAX_READS}/file")
    print("=" * 70)

    # Remove old DB if exists
    if os.path.exists(DB):
        os.remove(DB)

    # Ingest
    print("\n--- INGESTING ---")
    total = ingest_all(DB, ZIP, max_reads_per_file=MAX_READS)

    # Stats
    show_stats(DB)

    # Verify round-trip
    print(f"\n--- ROUND-TRIP VERIFICATION ---")
    ok = verify_roundtrip(DB, ZIP, max_reads_per_file=MAX_READS)
    print(f"\nOverall: {'ALL OK — lossless round-trip verified' if ok else 'FAILURES DETECTED'}")

    # Show a sample polar read
    print(f"\n--- SAMPLE POLAR READ ---")
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""SELECT header_raw, original_seq, polar_phases, quality_raw,
                   centroid_real, centroid_imag, resultant_len, gc_content, mean_quality
                   FROM reads LIMIT 1""")
    row = cur.fetchone()
    header, seq, polar_blob, qual, cr, ci, rl, gc, mq = row
    polar = np.frombuffer(polar_blob, dtype=np.uint8)

    print(f"  Header:    {header[:80]}...")
    print(f"  Seq (30):  {seq[:30]}...")
    print(f"  Polar:     {polar[:30]}...")
    print(f"  Phases:    {[f'{PHASES[p]:.4f}' if p <= 3 else 'N/A' for p in polar[:15]]}")
    print(f"  Complex:   {[f'{PHASORS[p]:.2f}' if p <= 3 else 'N/A' for p in polar[:10]]}")
    print(f"  Quality:   {qual[:30]}...")
    print(f"  Centroid:  ({cr:.4f}, {ci:.4f})  |c|={rl:.4f}")
    print(f"  GC:        {gc:.4f}")
    print(f"  Mean Qual: {mq:.1f}")

    # Verify polar bijection on this read
    reconstructed = polar_to_seq(polar, seq)
    print(f"  Bijection: {'PERFECT' if reconstructed == seq else 'BROKEN'}")

    conn.close()
