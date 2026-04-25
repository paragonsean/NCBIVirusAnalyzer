#!/usr/bin/env python3
"""
Count unique and duplicate DNA sequences in a FASTA file.

Usage:
  python fasta_uniqueness.py sequences.fasta
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path


DNA_RE = re.compile(r"[^ACGT]")


def parse_fasta_sequences(path: Path):
    seq_parts = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    yield "".join(seq_parts).upper()
                    seq_parts = []
            else:
                seq_parts.append(line)
    if seq_parts:
        yield "".join(seq_parts).upper()


def clean_dna(seq: str) -> str:
    return DNA_RE.sub("", seq)


def main() -> None:
    parser = argparse.ArgumentParser(description="Count duplicate and unique DNA sequences in FASTA.")
    parser.add_argument("input", help="Input FASTA file path.")
    parser.add_argument("--top", type=int, default=0, help="Show top N most repeated sequences.")
    args = parser.parse_args()

    fasta_path = Path(args.input)
    if not fasta_path.exists():
        raise FileNotFoundError(f"Input FASTA not found: {fasta_path}")

    counts = collections.Counter()
    total_sequences = 0

    for seq in parse_fasta_sequences(fasta_path):
        s = clean_dna(seq)
        if not s:
            continue
        counts[s] += 1
        total_sequences += 1

    unique_sequences = len(counts)
    duplicate_sequences = total_sequences - unique_sequences
    duplicate_rate = (100.0 * duplicate_sequences / total_sequences) if total_sequences else 0.0

    print(f"File: {fasta_path}")
    print(f"Total DNA sequences: {total_sequences:,}")
    print(f"Unique DNA sequences: {unique_sequences:,}")
    print(f"Duplicate DNA sequences: {duplicate_sequences:,}")
    print(f"Duplicate rate: {duplicate_rate:.2f}%")

    if args.top > 0:
        print(f"\nTop {args.top} repeated sequences:")
        repeated = [(seq, ct) for seq, ct in counts.items() if ct > 1]
        repeated.sort(key=lambda x: x[1], reverse=True)
        for i, (seq, ct) in enumerate(repeated[: args.top], start=1):
            print(f"{i:>2}. count={ct:,}  length={len(seq):,}")
            print(f"    {seq}")


if __name__ == "__main__":
    main()

