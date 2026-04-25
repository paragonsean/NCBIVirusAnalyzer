#!/usr/bin/env python3
"""
Build a duplicate-sequence dictionary from a FASTA file.

The output JSON groups identical cleaned DNA sequences together and keeps the
original FASTA identifiers for every occurrence.

Usage:
  python pattern_machine.py sequences.fasta --out sequences_duplicates.json
  python pattern_machine.py sequences.fasta --metadata-csv ml_training_data_full.csv
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any
from typing import Iterable

import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DNA_RE = re.compile(r"[^ACGT]")


def extract_fasta_identifier(header_line: str) -> str:
    """Return the first identifier token after '>'."""
    header = header_line.strip()
    if header.startswith(">"):
        header = header[1:]
    header = header.strip()
    if not header:
        return "unknown_id"
    return header.split()[0]


def clean_dna(seq: str) -> str:
    return DNA_RE.sub("", seq.upper())


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    return value


def load_metadata_by_accession(csv_path: Path) -> dict[str, dict[str, Any]]:
    df = pd.read_csv(csv_path, low_memory=False)
    if "Accession" not in df.columns:
        raise ValueError("Metadata CSV must contain an Accession column.")

    metadata: dict[str, dict[str, Any]] = {}
    for row in df.to_dict(orient="records"):
        accession = row.get("Accession")
        if pd.isna(accession):
            continue
        accession_key = str(accession)
        metadata[accession_key] = {
            key: clean_value(value)
            for key, value in row.items()
            if key != "Accession"
        }
    return metadata


def enrich_duplicate_data(data: dict, metadata_csv: Path) -> dict[str, int]:
    metadata_by_accession = load_metadata_by_accession(metadata_csv)
    sequence_entries = data.get("sequences", [])
    matched_accessions: set[str] = set()
    missing_accessions: set[str] = set()
    enriched_entries = 0

    for entry in sequence_entries:
        identifiers = entry.get("identifiers", [])
        entry_metadata: dict[str, dict[str, Any]] = {}

        for accession in identifiers:
            accession_key = str(accession)
            metadata = metadata_by_accession.get(accession_key)
            if metadata is None:
                missing_accessions.add(accession_key)
                continue
            entry_metadata[accession_key] = metadata
            matched_accessions.add(accession_key)

        entry["metadata_by_accession"] = entry_metadata
        if entry_metadata:
            enriched_entries += 1

    summary = {
        "metadata_rows": len(metadata_by_accession),
        "duplicate_sequence_entries": len(sequence_entries),
        "enriched_sequence_entries": enriched_entries,
        "matched_accessions": len(matched_accessions),
        "missing_accessions": len(missing_accessions),
    }
    data["metadata_source_file"] = str(metadata_csv.name)
    data["metadata_key"] = "Accession"
    data["metadata_summary"] = summary
    return summary


def parse_fasta_entries(path: Path, max_seqs: int | None = None) -> Iterable[tuple[str, str]]:
    """Yield (identifier, cleaned_sequence) records from FASTA."""
    current_id: str | None = None
    seq_parts: list[str] = []
    seen = 0

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue

            if line.startswith(">"):
                if current_id is not None and seq_parts:
                    yield current_id, clean_dna("".join(seq_parts))
                    seen += 1
                    if max_seqs is not None and seen >= max_seqs:
                        return
                current_id = extract_fasta_identifier(line)
                seq_parts = []
            else:
                seq_parts.append(line)

    if current_id is not None and seq_parts:
        yield current_id, clean_dna("".join(seq_parts))


def build_duplicate_sequence_json(
    fasta_path: Path,
    out_path: Path,
    metadata_csv: Path | None = None,
    max_seqs: int | None = None,
    show_progress: bool = False,
) -> dict:
    grouped: dict[str, list[str]] = collections.defaultdict(list)
    total_sequences = 0

    entries = parse_fasta_entries(fasta_path, max_seqs=max_seqs)
    if show_progress and tqdm is not None:
        entries = tqdm(entries, desc="Reading FASTA", unit="seq")

    for identifier, sequence in entries:
        if not sequence:
            continue
        grouped[sequence].append(identifier)
        total_sequences += 1

    sequence_rows = [
        {
            "sequence": sequence,
            "count": len(identifiers),
            "identifiers": identifiers,
        }
        for sequence, identifiers in grouped.items()
    ]
    sequence_rows.sort(key=lambda row: (row["count"], len(row["sequence"])), reverse=True)

    data = {
        "source_file": fasta_path.name,
        "max_seqs": max_seqs,
        "total_sequences": total_sequences,
        "unique_sequences": len(sequence_rows),
        "duplicate_sequences": total_sequences - len(sequence_rows),
        "sequences": sequence_rows,
    }

    metadata_summary = None
    if metadata_csv is not None:
        metadata_summary = enrich_duplicate_data(data, metadata_csv)

    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    if metadata_summary is not None:
        print(json.dumps(metadata_summary, indent=2))
    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create sequences_duplicates.json from identical FASTA DNA sequences."
    )
    parser.add_argument(
        "fasta",
        type=Path,
        nargs="?",
        default=Path("sequences.fasta"),
        help="Input FASTA path",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("sequences_duplicates.json"),
        help="Output duplicate JSON path",
    )
    parser.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Optional CSV with Accession metadata to attach to each identifier",
    )
    parser.add_argument("--max-seqs", type=int, default=None)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    data = build_duplicate_sequence_json(
        fasta_path=args.fasta,
        out_path=args.out,
        metadata_csv=args.metadata_csv,
        max_seqs=args.max_seqs,
        show_progress=args.progress,
    )
    print(f"Duplicate dictionary saved: {args.out}")
    print(f"Total sequences: {data['total_sequences']}")
    print(f"Unique sequences: {data['unique_sequences']}")
    print(f"Duplicate rows: {data['duplicate_sequences']}")


if __name__ == "__main__":
    main()
