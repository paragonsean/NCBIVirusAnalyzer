"""
Polar DNA Compressor — v2 (with real entropy coding)

Idea (adapted from PolarQuant, arXiv 2502.02617, originally for LLM KV-caches):
  • Map each base to a phase on the unit circle:  A=0, C=pi/2, G=pi, T=3pi/2
  • Treat a block of bases as a complex vector; sum -> a single complex number
      - magnitude  m  encodes base composition of the block
      - phase       theta encodes ordering information
  • Quantize (m, theta) to small ints, then run a real entropy coder
    (zlib / bz2 / lzma) on the byte stream.

We benchmark three things:
  (1) lossless polar encoding (phase-index per base, 2 bits/base, entropy-coded)
  (2) lossy block-polar encoding (m, theta per block)
  (3) plain 2-bit packed baseline + the same entropy coders.

Baseline to beat: 2 bits / base = 0.25 bytes/base.
"""

import numpy as np
import zlib, bz2, lzma, random, sys

BASES = "ACGT"
PHASES = {b: i * (np.pi / 2) for i, b in enumerate(BASES)}  # A=0, C=pi/2, G=pi, T=3pi/2
INV_PHASES = {v: k for k, v in PHASES.items()}

# ---------------- helpers ----------------

def to_complex(seq: str) -> np.ndarray:
    return np.exp(1j * np.array([PHASES[b] for b in seq]))

def phase_idx_stream(seq: str) -> np.ndarray:
    """Lossless: each base -> 0/1/2/3 (2 bits)."""
    return np.array([BASES.index(b) for b in seq], dtype=np.uint8)

def pack_2bit(seq: str) -> bytes:
    """Baseline: pack 4 bases per byte (2 bits each)."""
    idx = phase_idx_stream(seq)
    pad = (-len(idx)) % 4
    if pad:
        idx = np.concatenate([idx, np.zeros(pad, dtype=np.uint8)])
    packed = (idx[0::4] << 6) | (idx[1::4] << 4) | (idx[2::4] << 2) | idx[3::4]
    return packed.tobytes()

# ---------------- lossy block polar coder ----------------

def block_polar_encode(seq: str, block_size: int = 8,
                       mag_bits: int = 5, phase_bits: int = 8):
    """
    Sum complex-valued bases in blocks, quantize magnitude & phase.
    Lossy: blocks with the same (m, theta) collide.
    """
    z = to_complex(seq)
    pad = (-len(z)) % block_size
    if pad:
        z = np.concatenate([z, np.zeros(pad, dtype=complex)])
    blocks = z.reshape(-1, block_size).sum(axis=1)

    mag = np.abs(blocks)                   # in [0, block_size]
    phase = np.angle(blocks) % (2 * np.pi) # in [0, 2pi)

    mag_q = np.clip(np.round(mag * ((1 << mag_bits) - 1) / block_size),
                    0, (1 << mag_bits) - 1).astype(np.uint8)
    phase_q = np.clip(np.round(phase * ((1 << phase_bits) - 1) / (2 * np.pi)),
                      0, (1 << phase_bits) - 1).astype(np.uint8)
    return mag_q.tobytes() + phase_q.tobytes(), len(seq)

# ---------------- lossless polar (phase-index per base) ----------------

def lossless_polar_encode(seq: str) -> bytes:
    """Exact same info as 2-bit packing, but left as a byte/base stream
    so entropy coders can find higher-order redundancy."""
    return phase_idx_stream(seq).tobytes()

# ---------------- entropy coding wrappers ----------------

def measure(label: str, blob: bytes, n_bases: int, lossless: bool = True):
    sizes = {
        "raw":  len(blob),
        "zlib": len(zlib.compress(blob, 9)),
        "bz2":  len(bz2.compress(blob, 9)),
        "lzma": len(lzma.compress(blob, preset=9)),
    }
    print(f"\n{label}  ({'lossless' if lossless else 'LOSSY'})  input bytes = {len(blob)}")
    for k, v in sizes.items():
        bpb = 8 * v / n_bases
        ratio = (2.0) / bpb if bpb else float('inf')
        print(f"  {k:>5}: {v:>8} bytes  |  {bpb:5.3f} bits/base  |  {ratio:4.2f}x vs 2-bit raw")
    return sizes

# ---------------- test data generators ----------------

def random_dna(n, seed=0):
    rng = random.Random(seed)
    return "".join(rng.choices(BASES, k=n))

def skewed_dna(n, weights=(0.4, 0.1, 0.1, 0.4), seed=1):
    """AT-rich genome, like many bacterial sequences."""
    rng = random.Random(seed)
    return "".join(rng.choices(BASES, weights=weights, k=n))

def repeat_dna(n, motif="GATTACA"):
    return (motif * (n // len(motif) + 1))[:n]

# ---------------- integrity (lossless) ----------------

def roundtrip_lossless(seq: str) -> bool:
    idx = np.frombuffer(lossless_polar_encode(seq), dtype=np.uint8)
    recovered = "".join(BASES[i] for i in idx)
    return recovered == seq

# ---------------- main benchmark ----------------

def bench(seq: str, label: str):
    print("=" * 72)
    print(f"{label}  (len = {len(seq):,} bases)")
    print("=" * 72)

    assert roundtrip_lossless(seq), "lossless encode/decode mismatch"

    # 1. Baseline: plain 2-bit packed
    baseline = pack_2bit(seq)
    measure("2-bit packed baseline", baseline, len(seq), lossless=True)

    # 2. Lossless polar (1 byte/base phase stream)
    polar_bytes = lossless_polar_encode(seq)
    measure("Lossless polar (1 byte/base)", polar_bytes, len(seq), lossless=True)

    # 3. Lossy block-polar
    for bs in (4, 8, 16):
        blob, n = block_polar_encode(seq, block_size=bs, mag_bits=5, phase_bits=8)
        measure(f"Lossy block-polar  block={bs}  (5b mag + 8b phase)",
                blob, n, lossless=False)


if __name__ == "__main__":
    N = 50_000
    bench(random_dna(N),         "RANDOM DNA (max entropy)")
    bench(skewed_dna(N),         "AT-RICH DNA (bacterial-like)")
    bench(repeat_dna(N, "GATTACA"), "REPETITIVE DNA (GATTACA motif)")
