#!/usr/bin/env python3
"""
PatternMachine: mine and reuse repeating FASTA motifs.

Goal:
  - Find long repeated patterns (e.g. AAAGGGG, AAAGGGGG, ...)
  - Build a deterministic dictionary (X_0001 -> motif)
  - Translate FASTA sequences using longest-match tokenization
  - Reuse a saved dictionary later so you do not need to remine each run

Usage:
  python pattern_machine.py mine sequences.fasta --dict-out motifs.json
  python pattern_machine.py apply sequences.fasta --dict motifs.json --out translated.txt
  python pattern_machine.py report --dict motifs.json
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DNA_RE = re.compile(r"[^ACGT]")


def parse_fasta_sequences(path: Path) -> Iterable[str]:
    """Yield uppercased sequence strings from a FASTA file."""
    seq_parts: List[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if seq_parts:
                    yield "".join(seq_parts).upper()
                    seq_parts.clear()
            else:
                seq_parts.append(line)
    if seq_parts:
        yield "".join(seq_parts).upper()


def clean_dna(seq: str) -> str:
    return DNA_RE.sub("", seq)


@dataclass
class Candidate:
    pattern: str
    count: int
    score: int


def count_kmers(
    fasta_path: Path,
    min_len: int,
    max_len: int,
    min_count: int,
) -> Dict[str, int]:
    """Count k-mers for k in [min_len, max_len] across FASTA."""
    counters: Dict[int, collections.Counter] = {
        k: collections.Counter() for k in range(min_len, max_len + 1)
    }
    for seq in parse_fasta_sequences(fasta_path):
        s = clean_dna(seq)
        if not s:
            continue
        n = len(s)
        for k in range(min_len, max_len + 1):
            if n < k:
                continue
            c = counters[k]
            for i in range(n - k + 1):
                c[s[i : i + k]] += 1

    merged: Dict[str, int] = {}
    for k in range(min_len, max_len + 1):
        for pat, ct in counters[k].items():
            if ct >= min_count:
                merged[pat] = ct
    return merged


def tokenize_longest_match(seq: str, trie: Dict) -> List[str]:
    """Greedy longest-match tokenizer with fallback single bases."""
    out: List[str] = []
    i = 0
    n = len(seq)
    while i < n:
        node = trie
        j = i
        last_token = None
        last_j = i
        while j < n and seq[j] in node:
            node = node[seq[j]]
            j += 1
            token = node.get("_tok")
            if token is not None:
                last_token = token
                last_j = j
        if last_token is not None:
            out.append(last_token)
            i = last_j
        else:
            out.append(seq[i])
            i += 1
    return out


def build_trie(token_to_pattern: Dict[str, str]) -> Dict:
    trie: Dict = {}
    for token, pat in token_to_pattern.items():
        node = trie
        for ch in pat:
            node = node.setdefault(ch, {})
        node["_tok"] = token
    return trie


def estimate_encoded_units(fasta_path: Path, token_to_pattern: Dict[str, str]) -> int:
    """
    Unit cost estimator:
      - each base char costs 1 unit
      - each token occurrence costs 1 unit
    """
    trie = build_trie(token_to_pattern)
    total = 0
    for seq in parse_fasta_sequences(fasta_path):
        s = clean_dna(seq)
        if not s:
            continue
        total += len(tokenize_longest_match(s, trie))
    return total


def select_dictionary(
    fasta_path: Path,
    raw_counts: Dict[str, int],
    max_terms: int,
    candidate_pool: int,
) -> List[Candidate]:
    """
    Pick motifs that reduce encoded size under longest-match tokenization.
    Greedy forward selection from high-value candidate pool.
    """
    candidates: List[Candidate] = []
    for pat, ct in raw_counts.items():
        # Savings proxy before overlap effects:
        # each occurrence replaces len(pat) units with 1 token unit.
        gain = (len(pat) - 1) * ct
        if gain > 0:
            candidates.append(Candidate(pattern=pat, count=ct, score=gain))
    candidates.sort(key=lambda x: (x.score, len(x.pattern), x.count), reverse=True)
    candidates = candidates[:candidate_pool]

    chosen: List[Candidate] = []
    token_to_pattern: Dict[str, str] = {}
    baseline = estimate_encoded_units(fasta_path, token_to_pattern)

    for cand in candidates:
        if len(chosen) >= max_terms:
            break
        token = f"X_{len(chosen) + 1:04d}"
        trial = dict(token_to_pattern)
        trial[token] = cand.pattern
        new_units = estimate_encoded_units(fasta_path, trial)
        if new_units < baseline:
            token_to_pattern = trial
            chosen.append(cand)
            baseline = new_units

    return chosen


def mine_dictionary(
    fasta_path: Path,
    dict_out: Path,
    min_len: int,
    max_len: int,
    min_count: int,
    max_terms: int,
    candidate_pool: int,
) -> Dict:
    raw_counts = count_kmers(fasta_path, min_len, max_len, min_count)
    chosen = select_dictionary(fasta_path, raw_counts, max_terms, candidate_pool)

    dictionary = []
    for idx, c in enumerate(chosen, start=1):
        dictionary.append(
            {
                "token": f"X_{idx:04d}",
                "pattern": c.pattern,
                "count": c.count,
                "score": c.score,
                "length": len(c.pattern),
            }
        )

    out = {
        "format": "PatternMachine-v1",
        "source_file": str(fasta_path),
        "params": {
            "min_len": min_len,
            "max_len": max_len,
            "min_count": min_count,
            "max_terms": max_terms,
            "candidate_pool": candidate_pool,
        },
        "dictionary": dictionary,
    }
    dict_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def apply_dictionary(fasta_path: Path, dict_path: Path, out_path: Path) -> Dict[str, int]:
    data = json.loads(dict_path.read_text(encoding="utf-8"))
    token_to_pattern = {d["token"]: d["pattern"] for d in data["dictionary"]}
    trie = build_trie(token_to_pattern)

    total_bases = 0
    total_units = 0
    token_counts = collections.Counter()

    with out_path.open("w", encoding="utf-8") as out:
        for idx, seq in enumerate(parse_fasta_sequences(fasta_path), start=1):
            s = clean_dna(seq)
            if not s:
                continue
            tokens = tokenize_longest_match(s, trie)
            total_bases += len(s)
            total_units += len(tokens)
            for t in tokens:
                if t.startswith("X_"):
                    token_counts[t] += 1
            out.write(f">translated_{idx}\n")
            out.write(" ".join(tokens) + "\n")

    return {
        "total_bases": total_bases,
        "total_units": total_units,
        "dictionary_size": len(token_to_pattern),
        "token_uses": int(sum(token_counts.values())),
    }


def print_top_repeats(raw_counts: Dict[str, int], top_n: int = 15) -> None:
    ranked = sorted(raw_counts.items(), key=lambda x: (len(x[0]), x[1]), reverse=True)
    print("\nTop repeated motifs (longest first):")
    for pat, ct in ranked[:top_n]:
        print(f"  {pat:<24} {ct:>10}x")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine and apply repeated FASTA pattern dictionaries.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_mine = sub.add_parser("mine", help="Find repeated patterns and build dictionary.")
    p_mine.add_argument("input", help="Input FASTA file.")
    p_mine.add_argument("--dict-out", default="motif_dictionary.json", help="Output dictionary JSON.")
    p_mine.add_argument("--min-len", type=int, default=6)
    p_mine.add_argument("--max-len", type=int, default=12)
    p_mine.add_argument("--min-count", type=int, default=20)
    p_mine.add_argument("--max-terms", type=int, default=128)
    p_mine.add_argument("--candidate-pool", type=int, default=400)
    p_mine.add_argument("--show-top", type=int, default=15)

    p_apply = sub.add_parser("apply", help="Translate FASTA using existing dictionary.")
    p_apply.add_argument("input", help="Input FASTA file.")
    p_apply.add_argument("--dict", required=True, help="Dictionary JSON from mine step.")
    p_apply.add_argument("--out", default="translated_tokens.txt", help="Output translated text.")

    p_report = sub.add_parser("report", help="Show dictionary contents.")
    p_report.add_argument("--dict", required=True, help="Dictionary JSON.")
    p_report.add_argument("--top", type=int, default=30)

    args = parser.parse_args()

    if args.cmd == "mine":
        src = Path(args.input)
        raw_counts = count_kmers(src, args.min_len, args.max_len, args.min_count)
        print_top_repeats(raw_counts, top_n=args.show_top)
        result = mine_dictionary(
            fasta_path=src,
            dict_out=Path(args.dict_out),
            min_len=args.min_len,
            max_len=args.max_len,
            min_count=args.min_count,
            max_terms=args.max_terms,
            candidate_pool=args.candidate_pool,
        )
        print(f"\nDictionary saved: {args.dict_out}")
        print(f"Selected terms: {len(result['dictionary'])}")
        for item in result["dictionary"][: min(20, len(result["dictionary"]))]:
            print(f"  {item['token']} -> {item['pattern']}  (count={item['count']}, score={item['score']})")
        return

    if args.cmd == "apply":
        stats = apply_dictionary(Path(args.input), Path(args.dict), Path(args.out))
        ratio = stats["total_bases"] / max(stats["total_units"], 1)
        print(f"Translated output: {args.out}")
        print(
            f"Bases={stats['total_bases']:,}, Encoded units={stats['total_units']:,}, "
            f"Effective ratio={ratio:.2f}x, Token uses={stats['token_uses']:,}"
        )
        return

    if args.cmd == "report":
        data = json.loads(Path(args.dict).read_text(encoding="utf-8"))
        print(f"Dictionary: {args.dict}")
        print(f"Terms: {len(data.get('dictionary', []))}")
        for item in data.get("dictionary", [])[: args.top]:
            print(
                f"  {item['token']} -> {item['pattern']} "
                f"(len={item.get('length', len(item['pattern']))}, count={item.get('count', 'n/a')})"
            )


if __name__ == "__main__":
    main()

