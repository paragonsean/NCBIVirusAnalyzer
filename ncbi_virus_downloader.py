"""
NCBI Virus Sequence Downloader + Pattern Machine
A Tkinter desktop application with two tabs:
  1. Download — fetch virus sequences (FASTA / CSV) via the NCBI Datasets CLI
  2. Pattern Machine — find duplicate DNA sequences in a FASTA file and
     optionally enrich them with metadata from a CSV

On first run the program checks for the 'datasets' and 'dataformat'
binaries next to itself.  If missing it auto-downloads the correct
build for the current OS (Windows / macOS / Linux).
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import subprocess
import platform
import zipfile
import shutil
import urllib.request
import json
import csv
import collections
import re
import os
import sys
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────
# CLI binary download URLs (NCBI FTP)
# ──────────────────────────────────────────────────────────────

_BASE_FTP = "https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2"

_CLI_URLS = {
    # Windows (64-bit — covers AMD64 and x86_64 labels)
    ("Windows", "AMD64"):  (f"{_BASE_FTP}/win64/datasets.exe",
                            f"{_BASE_FTP}/win64/dataformat.exe"),
    ("Windows", "x86_64"): (f"{_BASE_FTP}/win64/datasets.exe",
                            f"{_BASE_FTP}/win64/dataformat.exe"),
    # macOS (single universal dir covers both Intel and Apple Silicon)
    ("Darwin", "x86_64"):  (f"{_BASE_FTP}/mac/datasets",
                            f"{_BASE_FTP}/mac/dataformat"),
    ("Darwin", "arm64"):   (f"{_BASE_FTP}/mac/datasets",
                            f"{_BASE_FTP}/mac/dataformat"),
    # Linux
    ("Linux", "x86_64"):   (f"{_BASE_FTP}/linux-amd64/datasets",
                            f"{_BASE_FTP}/linux-amd64/dataformat"),
    ("Linux", "aarch64"):  (f"{_BASE_FTP}/linux-arm64/datasets",
                            f"{_BASE_FTP}/linux-arm64/dataformat"),
}

# ──────────────────────────────────────────────────────────────
# Virus types: display name  →  taxon value for the CLI
# ──────────────────────────────────────────────────────────────

VIRUS_TYPES = {
    "Influenza A virus":       "11320",
    "Influenza A – H3N2":      "41857",
    "Influenza A – H1N1":      "114727",
    "Influenza A – H5N1":      "102793",
    "Influenza A – H7N9":      "1332244",
    "Influenza B virus":       "11520",
    "SARS-CoV-2":              "sars-cov-2",
    "MERS-CoV":                "1335626",
    "RSV (Respiratory Syncytial Virus)": "12814",
    "Ebola virus":             "186536",
    "Zika virus":              "64320",
    "Dengue virus":            "12637",
    "Mpox (Monkeypox)":        "10244",
    "HIV-1":                   "11676",
    "Hepatitis B virus":       "10407",
    "Hepatitis C virus":       "11103",
}

# ──────────────────────────────────────────────────────────────
# Locations (passed to --geo-location)
# ──────────────────────────────────────────────────────────────

LOCATIONS = [
    "Any Location",
    "USA", "United Kingdom", "China", "Japan", "India",
    "Brazil", "Australia", "Germany", "France", "South Africa",
    "Canada", "Mexico", "South Korea", "Italy", "Spain",
    "Africa", "Asia", "Europe", "North America",
    "South America", "Oceania",
]

# ──────────────────────────────────────────────────────────────
# Date helpers
# ──────────────────────────────────────────────────────────────

def _date_choices():
    return [
        "Any Date",
        "Last 30 days", "Last 90 days", "Last 6 months",
        "Last 1 year", "Last 2 years", "Last 5 years",
    ]

def _resolve_date(label):
    mapping = {
        "Last 30 days": 30, "Last 90 days": 90, "Last 6 months": 180,
        "Last 1 year": 365, "Last 2 years": 730, "Last 5 years": 1825,
    }
    days = mapping.get(label)
    if days is None:
        return None
    return (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

# CSV fields that dataformat will export
CSV_FIELDS = [
    "accession", "virus-name", "virus-tax-id",
    "isolate-collection-date", "geo-location",
    "host-name", "length", "completeness",
    "release-date", "sourcedb",
]

# ──────────────────────────────────────────────────────────────
# Helpers: locate / download CLI binaries
# ──────────────────────────────────────────────────────────────

def _app_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def _binary_name(tool):
    ext = ".exe" if platform.system() == "Windows" else ""
    return tool + ext

def _find_binary(tool):
    local = os.path.join(_app_dir(), _binary_name(tool))
    if os.path.isfile(local):
        return local
    return shutil.which(_binary_name(tool))

def _download_url_for(tool_index):
    system = platform.system()
    machine = platform.machine()
    key = (system, machine)
    if key not in _CLI_URLS:
        raise RuntimeError(
            f"Unsupported platform: {system} {machine}. "
            f"Supported: {list(_CLI_URLS.keys())}"
        )
    return _CLI_URLS[key][tool_index]

def _download_binary(tool, tool_index, progress_callback=None):
    url = _download_url_for(tool_index)
    dest = os.path.join(_app_dir(), _binary_name(tool))
    if progress_callback:
        progress_callback(f"Downloading {tool} from NCBI…")
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=120) as resp:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(256 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if progress_callback and total > 0:
                    pct = downloaded / total * 100
                    progress_callback(
                        f"Downloading {tool}… {downloaded // 1024:,} KB "
                        f"/ {total // 1024:,} KB ({pct:.0f}%)"
                    )
    if platform.system() != "Windows":
        os.chmod(dest, 0o755)
    return dest

def ensure_cli(progress_callback=None):
    tools = {}
    for idx, name in enumerate(("datasets", "dataformat")):
        path = _find_binary(name)
        if path is None:
            path = _download_binary(name, idx, progress_callback)
        tools[name] = path
    return tools

# ──────────────────────────────────────────────────────────────
# Core download logic using CLI
# ──────────────────────────────────────────────────────────────

def run_datasets_download(cli_paths, taxon, geo_location, released_after,
                          output_zip, complete_only=False, log_callback=None):
    cmd = [
        cli_paths["datasets"], "download", "virus", "genome",
        "taxon", taxon,
        "--filename", output_zip,
        "--no-progressbar",
    ]
    if geo_location and geo_location != "Any Location":
        cmd += ["--geo-location", geo_location]
    if released_after:
        cmd += ["--released-after", released_after]
    if complete_only:
        cmd += ["--complete-only"]
    if log_callback:
        log_callback(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"datasets CLI error:\n{err}")
    if not os.path.isfile(output_zip):
        raise FileNotFoundError("Download produced no output file.")
    return output_zip

def extract_fasta(zip_path, dest_fasta):
    fasta_member = "ncbi_dataset/data/genomic.fna"
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        if fasta_member not in names:
            raise FileNotFoundError(
                f"No FASTA found in download. Archive contains: {names}\n"
                "The query may have returned 0 sequences — try broader filters."
            )
        with z.open(fasta_member) as src, open(dest_fasta, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return dest_fasta

def extract_csv(cli_paths, zip_path, dest_csv):
    jsonl_member = "ncbi_dataset/data/data_report.jsonl"
    with zipfile.ZipFile(zip_path) as z:
        if jsonl_member not in z.namelist():
            raise FileNotFoundError(
                "No data report found in download. "
                "The query may have returned 0 sequences — try broader filters."
            )
    cmd = [
        cli_paths["dataformat"], "tsv", "virus-genome",
        "--package", zip_path,
        "--fields", ",".join(CSV_FIELDS),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        err = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"dataformat error:\n{err}")
    lines = result.stdout.strip().split("\n")
    with open(dest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line.split("\t"))
    return dest_csv


# ══════════════════════════════════════════════════════════════
# Pattern Machine  — duplicate-sequence detection (embedded)
# ══════════════════════════════════════════════════════════════

DNA_RE = re.compile(r"[^ACGT]")

def _clean_dna(seq):
    return DNA_RE.sub("", seq.upper())

def _extract_fasta_id(header_line):
    header = header_line.strip().lstrip(">").strip()
    return header.split()[0] if header else "unknown_id"

def parse_fasta_entries(path, max_seqs=None):
    """Yield (identifier, cleaned_sequence) from a FASTA file."""
    current_id = None
    seq_parts = []
    seen = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None and seq_parts:
                    yield current_id, _clean_dna("".join(seq_parts))
                    seen += 1
                    if max_seqs is not None and seen >= max_seqs:
                        return
                current_id = _extract_fasta_id(line)
                seq_parts = []
            else:
                seq_parts.append(line)
    if current_id is not None and seq_parts:
        yield current_id, _clean_dna("".join(seq_parts))

def _clean_csv_value(value):
    """Return a JSON-safe value (handle NaN, numpy scalars, etc.)."""
    try:
        import pandas as pd
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        return value.item()
    return value

def load_metadata_by_accession(csv_path):
    """Read CSV into {accession: {col: val, …}} dict."""
    import pandas as pd
    df = pd.read_csv(csv_path, low_memory=False)
    if "Accession" not in df.columns:
        raise ValueError("Metadata CSV must contain an 'Accession' column.")
    metadata = {}
    for row in df.to_dict(orient="records"):
        acc = row.get("Accession")
        try:
            import pandas as _pd
            if _pd.isna(acc):
                continue
        except Exception:
            if acc is None:
                continue
        metadata[str(acc)] = {
            k: _clean_csv_value(v) for k, v in row.items() if k != "Accession"
        }
    return metadata

def build_duplicate_dict(fasta_path, out_path,
                         metadata_csv=None, max_seqs=None,
                         progress_callback=None):
    """Build JSON of grouped duplicate sequences. Returns summary dict."""
    grouped = collections.defaultdict(list)
    total = 0

    for identifier, sequence in parse_fasta_entries(fasta_path, max_seqs):
        if not sequence:
            continue
        grouped[sequence].append(identifier)
        total += 1
        if progress_callback and total % 500 == 0:
            progress_callback(f"Parsed {total:,} sequences…")

    rows = [
        {"sequence": seq, "count": len(ids), "identifiers": ids}
        for seq, ids in grouped.items()
    ]
    rows.sort(key=lambda r: (r["count"], len(r["sequence"])), reverse=True)

    data = {
        "source_file": os.path.basename(fasta_path),
        "max_seqs": max_seqs,
        "total_sequences": total,
        "unique_sequences": len(rows),
        "duplicate_sequences": total - len(rows),
        "sequences": rows,
    }

    # Enrich with metadata CSV if provided
    meta_summary = None
    if metadata_csv:
        if progress_callback:
            progress_callback("Enriching with metadata…")
        meta_by_acc = load_metadata_by_accession(metadata_csv)
        enriched = 0
        matched_accs = set()
        missing_accs = set()
        for entry in rows:
            entry_meta = {}
            for acc in entry["identifiers"]:
                m = meta_by_acc.get(str(acc))
                if m is None:
                    missing_accs.add(acc)
                else:
                    entry_meta[acc] = m
                    matched_accs.add(acc)
            entry["metadata_by_accession"] = entry_meta
            if entry_meta:
                enriched += 1
        meta_summary = {
            "metadata_rows": len(meta_by_acc),
            "enriched_entries": enriched,
            "matched_accessions": len(matched_accs),
            "missing_accessions": len(missing_accs),
        }
        data["metadata_source_file"] = os.path.basename(metadata_csv)
        data["metadata_summary"] = meta_summary

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return {
        "total_sequences": total,
        "unique_sequences": len(rows),
        "duplicate_sequences": total - len(rows),
        "metadata": meta_summary,
    }


# ══════════════════════════════════════════════════════════════
# Tkinter Application  (tabbed: Download | Pattern Machine)
# ══════════════════════════════════════════════════════════════

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NCBI Virus Tools")
        self.geometry("930x760")
        self.resizable(False, False)
        self.configure(bg="#f5f6fa")

        self._cli_paths = None

        # ── Header ──
        header = tk.Frame(self, bg="#2c3e50", height=54)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(header, text="NCBI Virus Tools",
                 font=("Segoe UI", 15, "bold"),
                 fg="white", bg="#2c3e50").pack(pady=12)

        # ── Tabs ──
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        self._build_download_tab()
        self._build_pattern_tab()
        self._build_forecast_tab()
        self._build_growth_tab()

        # Kick off CLI check
        threading.Thread(target=self._ensure_cli_thread, daemon=True).start()

    # ────────────────────────────────────────────────────────
    #  Shared helpers
    # ────────────────────────────────────────────────────────
    def _make_log(self, parent):
        """Create and return a log Text widget inside parent."""
        tk.Label(parent, text="Log", font=("Segoe UI", 9, "bold"),
                 bg="#f5f6fa", anchor="w").pack(anchor="w", padx=4, pady=(6, 1))
        log = tk.Text(parent, height=7, font=("Consolas", 9),
                      bg="#ecf0f1", relief="flat", state="disabled")
        log.pack(fill="x", padx=4, pady=(0, 4))
        return log

    def _log_to(self, log_widget, msg):
        def _do():
            log_widget.config(state="normal")
            log_widget.insert("end",
                f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            log_widget.see("end")
            log_widget.config(state="disabled")
        self.after(0, _do)

    # ────────────────────────────────────────────────────────
    #  TAB 1 — Download
    # ────────────────────────────────────────────────────────
    def _build_download_tab(self):
        tab = tk.Frame(self.notebook, bg="#f5f6fa")
        self.notebook.add(tab, text="  Download  ")

        form = tk.Frame(tab, bg="#f5f6fa", padx=24, pady=14)
        form.pack(fill="both", expand=True)

        r = 0
        # Virus
        tk.Label(form, text="Virus Type", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        self.dl_virus_var = tk.StringVar(value=list(VIRUS_TYPES.keys())[0])
        ttk.Combobox(form, textvariable=self.dl_virus_var,
                     values=list(VIRUS_TYPES.keys()),
                     state="readonly", width=50, font=("Segoe UI", 10)
                     ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)); r += 1

        # Location
        tk.Label(form, text="Location", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        self.dl_location_var = tk.StringVar(value="Any Location")
        ttk.Combobox(form, textvariable=self.dl_location_var,
                     values=LOCATIONS,
                     state="readonly", width=50, font=("Segoe UI", 10)
                     ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)); r += 1

        # Date
        tk.Label(form, text="Released After", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        self.dl_date_var = tk.StringVar(value="Any Date")
        ttk.Combobox(form, textvariable=self.dl_date_var,
                     values=_date_choices(),
                     state="readonly", width=50, font=("Segoe UI", 10)
                     ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)); r += 1

        # Complete only
        self.dl_complete_var = tk.BooleanVar(value=False)
        tk.Checkbutton(form, text="Complete sequences only",
                       variable=self.dl_complete_var,
                       font=("Segoe UI", 10), bg="#f5f6fa"
                       ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        # Buttons
        bf = tk.Frame(form, bg="#f5f6fa")
        bf.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 6)); r += 1

        self.dl_fasta_btn = tk.Button(
            bf, text="Download FASTA", font=("Segoe UI", 10, "bold"),
            bg="#27ae60", fg="white", padx=16, pady=5, relief="flat",
            cursor="hand2", command=lambda: self._on_dl("fasta"))
        self.dl_fasta_btn.pack(side="left", padx=(0, 8))

        self.dl_csv_btn = tk.Button(
            bf, text="Download CSV", font=("Segoe UI", 10, "bold"),
            bg="#e67e22", fg="white", padx=16, pady=5, relief="flat",
            cursor="hand2", command=lambda: self._on_dl("csv"))
        self.dl_csv_btn.pack(side="left", padx=(0, 8))

        self.dl_both_btn = tk.Button(
            bf, text="Download Both", font=("Segoe UI", 10, "bold"),
            bg="#3498db", fg="white", padx=16, pady=5, relief="flat",
            cursor="hand2", command=lambda: self._on_dl("both"))
        self.dl_both_btn.pack(side="left")

        # Progress
        self.dl_progress = ttk.Progressbar(form, length=670, mode="indeterminate")
        self.dl_progress.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 2)); r += 1

        # Status
        self.dl_status_var = tk.StringVar(value="Checking for NCBI Datasets CLI…")
        tk.Label(form, textvariable=self.dl_status_var,
                 font=("Segoe UI", 9), bg="#f5f6fa", fg="#555",
                 anchor="w", wraplength=670
                 ).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        # Log
        log_frame = tk.Frame(form, bg="#f5f6fa")
        log_frame.grid(row=r, column=0, columnspan=2, sticky="we")
        self.dl_log = self._make_log(log_frame)

        self._dl_buttons = (self.dl_fasta_btn, self.dl_csv_btn, self.dl_both_btn)
        self._dl_set_enabled(False)

    def _dl_set_enabled(self, enabled):
        s = "normal" if enabled else "disabled"
        for b in self._dl_buttons:
            b.config(state=s)

    # ── CLI bootstrap ──
    def _ensure_cli_thread(self):
        try:
            self._log_to(self.dl_log,
                         f"Platform: {platform.system()} {platform.machine()}")
            self._log_to(self.dl_log, f"App directory: {_app_dir()}")

            def prog(msg):
                self.after(0, lambda: self.dl_status_var.set(msg))
                self._log_to(self.dl_log, msg)

            self._cli_paths = ensure_cli(progress_callback=prog)
            self._log_to(self.dl_log,
                         f"datasets: {self._cli_paths['datasets']}")
            self._log_to(self.dl_log,
                         f"dataformat: {self._cli_paths['dataformat']}")

            r = subprocess.run(
                [self._cli_paths["datasets"], "--version"],
                capture_output=True, text=True, timeout=10)
            self._log_to(self.dl_log, f"CLI version: {r.stdout.strip()}")

            self.after(0, lambda: self.dl_status_var.set(
                "Ready. Select options and click a Download button."))
            self.after(0, lambda: self._dl_set_enabled(True))
        except Exception as e:
            self._log_to(self.dl_log, f"CLI setup error: {e}")
            self.after(0, lambda: self.dl_status_var.set(
                f"Error setting up CLI: {e}"))

    # ── Download action ──
    def _on_dl(self, fmt):
        if not self._cli_paths:
            messagebox.showwarning("Not Ready",
                "NCBI Datasets CLI is still being set up.")
            return

        virus_label = self.dl_virus_var.get().replace(" ", "_").replace("–", "")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"ncbi_{virus_label}_{stamp}"

        if fmt == "fasta":
            p = filedialog.asksaveasfilename(
                title="Save FASTA", defaultextension=".fasta",
                initialfile=base + ".fasta",
                filetypes=[("FASTA", "*.fasta *.fna"), ("All", "*.*")])
            if not p: return
            paths = {"fasta": p}
        elif fmt == "csv":
            p = filedialog.asksaveasfilename(
                title="Save CSV", defaultextension=".csv",
                initialfile=base + ".csv",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")])
            if not p: return
            paths = {"csv": p}
        else:
            folder = filedialog.askdirectory(title="Choose folder for FASTA + CSV")
            if not folder: return
            paths = {"fasta": os.path.join(folder, base + ".fasta"),
                     "csv":   os.path.join(folder, base + ".csv")}

        self._dl_set_enabled(False)
        self.dl_progress.start(15)
        threading.Thread(target=self._dl_thread,
                         args=(fmt, paths), daemon=True).start()

    def _dl_thread(self, fmt, paths):
        tmp_zip = None
        try:
            virus = self.dl_virus_var.get()
            taxon = VIRUS_TYPES[virus]
            location = self.dl_location_var.get()
            released_after = _resolve_date(self.dl_date_var.get())
            complete = self.dl_complete_var.get()

            self._log_to(self.dl_log,
                         f"Virus: {virus} (taxon {taxon}) | "
                         f"{location} | after {released_after or 'any'}")
            self.after(0, lambda: self.dl_status_var.set(
                "Downloading from NCBI (this may take a moment)…"))

            tmp_zip = os.path.join(tempfile.gettempdir(),
                                   f"ncbi_dl_{os.getpid()}.zip")
            run_datasets_download(
                self._cli_paths, taxon, location, released_after,
                tmp_zip, complete_only=complete,
                log_callback=lambda m: self._log_to(self.dl_log, m))

            with zipfile.ZipFile(tmp_zip) as z:
                self._log_to(self.dl_log, f"Zip contents: {z.namelist()}")

            if "fasta" in paths:
                self.after(0, lambda: self.dl_status_var.set("Extracting FASTA…"))
                extract_fasta(tmp_zip, paths["fasta"])
                kb = os.path.getsize(paths["fasta"]) / 1024
                self._log_to(self.dl_log,
                             f"FASTA saved: {paths['fasta']} ({kb:,.0f} KB)")

            if "csv" in paths:
                self.after(0, lambda: self.dl_status_var.set(
                    "Converting metadata to CSV…"))
                extract_csv(self._cli_paths, tmp_zip, paths["csv"])
                kb = os.path.getsize(paths["csv"]) / 1024
                self._log_to(self.dl_log,
                             f"CSV saved: {paths['csv']} ({kb:,.0f} KB)")

            saved = " and ".join(os.path.basename(p) for p in paths.values())
            self.after(0, lambda: self.dl_status_var.set(f"Done! Saved: {saved}"))
            self.after(0, lambda: messagebox.showinfo(
                "Download Complete",
                "Files saved:\n" + "\n".join(paths.values())))

        except FileNotFoundError as e:
            err = str(e)
            self._log_to(self.dl_log, f"No data: {e}")
            self.after(0, lambda: self.dl_status_var.set(
                "No sequences found — try broader filters."))
            self.after(0, lambda: messagebox.showwarning("No Sequences", err))
        except Exception as e:
            err = str(e)
            self._log_to(self.dl_log, f"Error: {err}")
            self.after(0, lambda: self.dl_status_var.set(f"Failed: {err}"))
            self.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            if tmp_zip and os.path.isfile(tmp_zip):
                try: os.remove(tmp_zip)
                except OSError: pass
            self.after(0, lambda: self.dl_progress.stop())
            self.after(0, lambda: self._dl_set_enabled(True))

    # ────────────────────────────────────────────────────────
    #  TAB 2 — Pattern Machine
    # ────────────────────────────────────────────────────────
    def _build_pattern_tab(self):
        tab = tk.Frame(self.notebook, bg="#f5f6fa")
        self.notebook.add(tab, text="  Pattern Machine  ")

        form = tk.Frame(tab, bg="#f5f6fa", padx=24, pady=14)
        form.pack(fill="both", expand=True)

        r = 0

        # FASTA input
        tk.Label(form, text="FASTA Input File", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1

        fasta_row = tk.Frame(form, bg="#f5f6fa")
        fasta_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)); r += 1
        self.pm_fasta_var = tk.StringVar()
        tk.Entry(fasta_row, textvariable=self.pm_fasta_var,
                 width=55, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        tk.Button(fasta_row, text="Browse…", font=("Segoe UI", 9),
                  command=self._pm_browse_fasta).pack(side="left")

        # Optional metadata CSV
        tk.Label(form, text="Metadata CSV (optional)", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1

        csv_row = tk.Frame(form, bg="#f5f6fa")
        csv_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)); r += 1
        self.pm_csv_var = tk.StringVar()
        tk.Entry(csv_row, textvariable=self.pm_csv_var,
                 width=55, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        tk.Button(csv_row, text="Browse…", font=("Segoe UI", 9),
                  command=self._pm_browse_csv).pack(side="left")

        self.pm_embed_metadata_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            form,
            text="Embed metadata in duplicates JSON (larger file; not needed for training)",
            variable=self.pm_embed_metadata_var,
            font=("Segoe UI", 10),
            bg="#f5f6fa"
        ).grid(row=r, column=0, sticky="w", pady=(0, 10)); r += 1

        # Max sequences
        tk.Label(form, text="Max Sequences (blank = all)",
                 font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        self.pm_max_var = tk.StringVar()
        tk.Entry(form, textvariable=self.pm_max_var,
                 width=15, font=("Segoe UI", 10)
                 ).grid(row=r, column=0, sticky="w", pady=(0, 14)); r += 1

        # Run button
        bf = tk.Frame(form, bg="#f5f6fa")
        bf.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 6)); r += 1

        self.pm_run_btn = tk.Button(
            bf, text="Find Duplicates", font=("Segoe UI", 10, "bold"),
            bg="#8e44ad", fg="white", padx=20, pady=5, relief="flat",
            cursor="hand2", command=self._on_pm_run)
        self.pm_run_btn.pack(side="left")

        # Progress
        self.pm_progress = ttk.Progressbar(form, length=670, mode="indeterminate")
        self.pm_progress.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 2)); r += 1

        # Status
        self.pm_status_var = tk.StringVar(
            value="Select a FASTA file and click Find Duplicates.")
        tk.Label(form, textvariable=self.pm_status_var,
                 font=("Segoe UI", 9), bg="#f5f6fa", fg="#555",
                 anchor="w", wraplength=670
                 ).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        # Results summary
        self.pm_result_text = tk.Text(form, height=5, font=("Consolas", 10),
                                      bg="#ecf0f1", relief="flat",
                                      state="disabled")
        self.pm_result_text.grid(row=r, column=0, columnspan=2,
                                 sticky="we", pady=(6, 2)); r += 1

        # Log
        log_frame = tk.Frame(form, bg="#f5f6fa")
        log_frame.grid(row=r, column=0, columnspan=2, sticky="we")
        self.pm_log = self._make_log(log_frame)

    def _pm_browse_fasta(self):
        p = filedialog.askopenfilename(
            title="Select FASTA file",
            filetypes=[("FASTA", "*.fasta *.fna *.fa"), ("All", "*.*")])
        if p:
            self.pm_fasta_var.set(p)

    def _pm_browse_csv(self):
        p = filedialog.askopenfilename(
            title="Select metadata CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.pm_csv_var.set(p)

    def _on_pm_run(self):
        fasta = self.pm_fasta_var.get().strip()
        if not fasta or not os.path.isfile(fasta):
            messagebox.showwarning("No FASTA",
                "Please select a valid FASTA file.")
            return

        # Choose output path
        default_out = os.path.splitext(fasta)[0] + "_duplicates.json"
        out_path = filedialog.asksaveasfilename(
            title="Save duplicates JSON",
            defaultextension=".json",
            initialfile=os.path.basename(default_out),
            initialdir=os.path.dirname(fasta),
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if not out_path:
            return

        csv_path = self.pm_csv_var.get().strip()
        embed_metadata = self.pm_embed_metadata_var.get()
        if embed_metadata and (not csv_path or not os.path.isfile(csv_path)):
            messagebox.showwarning("No Metadata CSV",
                "Select a valid metadata CSV or uncheck metadata embedding.")
            return
        meta_csv = csv_path if embed_metadata else None
        max_str = self.pm_max_var.get().strip()
        max_seqs = int(max_str) if max_str.isdigit() else None

        self.pm_run_btn.config(state="disabled")
        self.pm_progress.start(15)
        threading.Thread(target=self._pm_thread,
                         args=(fasta, out_path, meta_csv, max_seqs),
                         daemon=True).start()

    def _pm_thread(self, fasta, out_path, meta_csv, max_seqs):
        try:
            self._log_to(self.pm_log, f"FASTA: {fasta}")
            if meta_csv:
                self._log_to(self.pm_log, f"Metadata CSV: {meta_csv}")
            self.after(0, lambda: self.pm_status_var.set(
                "Scanning for duplicate sequences…"))

            def prog(msg):
                self.after(0, lambda: self.pm_status_var.set(msg))
                self._log_to(self.pm_log, msg)

            result = build_duplicate_dict(
                fasta, out_path,
                metadata_csv=meta_csv,
                max_seqs=max_seqs,
                progress_callback=prog)

            # Show summary
            summary_lines = [
                f"Total sequences:     {result['total_sequences']:,}",
                f"Unique sequences:    {result['unique_sequences']:,}",
                f"Duplicate rows:      {result['duplicate_sequences']:,}",
            ]
            if result.get("metadata"):
                m = result["metadata"]
                summary_lines += [
                    f"Metadata matched:    {m['matched_accessions']:,} accessions",
                    f"Metadata missing:    {m['missing_accessions']:,} accessions",
                ]

            def _show():
                self.pm_result_text.config(state="normal")
                self.pm_result_text.delete("1.0", "end")
                self.pm_result_text.insert("end", "\n".join(summary_lines))
                self.pm_result_text.config(state="disabled")
            self.after(0, _show)

            self._log_to(self.pm_log, f"Saved: {out_path}")
            self.after(0, lambda: self.pm_status_var.set(
                f"Done! Saved to {os.path.basename(out_path)}"))
            self.after(0, lambda: messagebox.showinfo(
                "Pattern Machine Complete",
                f"Duplicate analysis saved to:\n{out_path}\n\n"
                + "\n".join(summary_lines)))

        except Exception as e:
            err = str(e)
            self._log_to(self.pm_log, f"Error: {err}")
            self.after(0, lambda: self.pm_status_var.set(f"Error: {err}"))
            self.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            self.after(0, lambda: self.pm_progress.stop())
            self.after(0, lambda: self.pm_run_btn.config(state="normal"))

    # ────────────────────────────────────────────────────────
    #  TAB 3 — Strain Forecast
    # ────────────────────────────────────────────────────────
    def _build_forecast_tab(self):
        tab = tk.Frame(self.notebook, bg="#f5f6fa")
        self.notebook.add(tab, text="  Strain Forecast  ")

        form = tk.Frame(tab, bg="#f5f6fa", padx=24, pady=14)
        form.pack(fill="both", expand=True)

        r = 0

        tk.Label(form, text="Metadata CSV", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        csv_row = tk.Frame(form, bg="#f5f6fa")
        csv_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        self.fc_csv_var = tk.StringVar()
        tk.Entry(csv_row, textvariable=self.fc_csv_var,
                 width=58, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        tk.Button(csv_row, text="Browse…", font=("Segoe UI", 9),
                  command=self._fc_browse_csv).pack(side="left")

        tk.Label(form, text="Duplicates JSON", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        json_row = tk.Frame(form, bg="#f5f6fa")
        json_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        self.fc_json_var = tk.StringVar()
        tk.Entry(json_row, textvariable=self.fc_json_var,
                 width=58, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        tk.Button(json_row, text="Browse…", font=("Segoe UI", 9),
                  command=self._fc_browse_json).pack(side="left")

        tk.Label(form, text="Output Forecast CSV", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        out_row = tk.Frame(form, bg="#f5f6fa")
        out_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        self.fc_out_var = tk.StringVar(value=os.path.abspath(
            "global_outbreak_forecast_2020_2025.csv"))
        tk.Entry(out_row, textvariable=self.fc_out_var,
                 width=58, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        tk.Button(out_row, text="Browse…", font=("Segoe UI", 9),
                  command=self._fc_browse_output).pack(side="left")

        tk.Label(form, text="Countries (comma-separated)",
                 font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        self.fc_countries_var = tk.StringVar(
            value="USA, Canada, France, Mexico, United Kingdom")
        tk.Entry(form, textvariable=self.fc_countries_var,
                 width=68, font=("Segoe UI", 10)
                 ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1

        opts = tk.Frame(form, bg="#f5f6fa")
        opts.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 10)); r += 1

        self.fc_years_var = tk.StringVar(value="2020-2025")
        self.fc_training_years_var = tk.StringVar(value="2020-2024")
        self.fc_backtest_var = tk.StringVar(value="2025")
        self.fc_dominant_year_var = tk.StringVar(value="2025")
        self.fc_epochs_var = tk.StringVar(value="10")
        self.fc_batch_var = tk.StringVar(value="256")
        self.fc_topn_var = tk.StringVar(value="3")

        fields = [
            ("Forecast Years", self.fc_years_var, 12),
            ("Training Years", self.fc_training_years_var, 12),
            ("Backtest Year", self.fc_backtest_var, 8),
            ("Dominant Year", self.fc_dominant_year_var, 8),
            ("Epochs", self.fc_epochs_var, 6),
            ("Batch Size", self.fc_batch_var, 8),
            ("Top N", self.fc_topn_var, 5),
        ]
        for label, var, width in fields:
            cell = tk.Frame(opts, bg="#f5f6fa")
            cell.pack(side="left", padx=(0, 12))
            tk.Label(cell, text=label, font=("Segoe UI", 9, "bold"),
                     bg="#f5f6fa").pack(anchor="w")
            tk.Entry(cell, textvariable=var, width=width,
                     font=("Segoe UI", 10)).pack(anchor="w")

        self.fc_run_btn = tk.Button(
            form, text="Run Strain Forecast", font=("Segoe UI", 10, "bold"),
            bg="#16a085", fg="white", padx=20, pady=5, relief="flat",
            cursor="hand2", command=self._on_fc_run)
        self.fc_run_btn.grid(row=r, column=0, sticky="w", pady=(2, 6)); r += 1

        self.fc_progress_var = tk.DoubleVar(value=0)
        self.fc_progress = ttk.Progressbar(
            form,
            length=690,
            mode="determinate",
            maximum=100,
            variable=self.fc_progress_var)
        self.fc_progress.grid(row=r, column=0, columnspan=2, sticky="w", pady=(4, 2)); r += 1

        self.fc_status_var = tk.StringVar(
            value="Select a metadata CSV and duplicates JSON, then run forecast.")
        tk.Label(form, textvariable=self.fc_status_var,
                 font=("Segoe UI", 9), bg="#f5f6fa", fg="#555",
                 anchor="w", wraplength=690
                 ).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        self.fc_result_text = tk.Text(form, height=6, font=("Consolas", 10),
                                      bg="#ecf0f1", relief="flat",
                                      state="disabled")
        self.fc_result_text.grid(row=r, column=0, columnspan=2,
                                 sticky="we", pady=(6, 2)); r += 1

        log_frame = tk.Frame(form, bg="#f5f6fa")
        log_frame.grid(row=r, column=0, columnspan=2, sticky="we")
        self.fc_log = self._make_log(log_frame)

    def _fc_browse_csv(self):
        p = filedialog.askopenfilename(
            title="Select metadata CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.fc_csv_var.set(p)

    def _fc_browse_json(self):
        p = filedialog.askopenfilename(
            title="Select duplicates JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")])
        if p:
            self.fc_json_var.set(p)

    def _fc_browse_output(self):
        p = filedialog.asksaveasfilename(
            title="Save forecast CSV",
            defaultextension=".csv",
            initialfile=os.path.basename(self.fc_out_var.get() or "forecast.csv"),
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.fc_out_var.set(p)

    def _on_fc_run(self):
        metadata_csv = self.fc_csv_var.get().strip()
        duplicates_json = self.fc_json_var.get().strip()
        output_csv = self.fc_out_var.get().strip()

        if not metadata_csv or not os.path.isfile(metadata_csv):
            messagebox.showwarning("No Metadata CSV",
                "Please select a valid metadata CSV file.")
            return
        if not duplicates_json or not os.path.isfile(duplicates_json):
            messagebox.showwarning("No Duplicates JSON",
                "Please select a valid duplicates JSON file.")
            return
        if not output_csv:
            messagebox.showwarning("No Output CSV",
                "Please choose where to save the forecast CSV.")
            return

        try:
            countries = [c.strip() for c in self.fc_countries_var.get().split(",")
                         if c.strip()]
            years_text = self.fc_years_var.get().strip()
            training_years_text = self.fc_training_years_var.get().strip()
            backtest_year = int(self.fc_backtest_var.get().strip())
            dominant_year = int(self.fc_dominant_year_var.get().strip())
            epochs = int(self.fc_epochs_var.get().strip())
            batch_size = int(self.fc_batch_var.get().strip())
            top_n = int(self.fc_topn_var.get().strip())
            if not countries:
                raise ValueError("Enter at least one country.")
            if epochs < 1 or batch_size < 1 or top_n < 1:
                raise ValueError("Epochs, batch size, and top N must be positive.")
        except ValueError as e:
            messagebox.showwarning("Invalid Forecast Options", str(e))
            return

        self.fc_run_btn.config(state="disabled")
        self.fc_progress_var.set(0)
        threading.Thread(
            target=self._fc_thread,
            args=(metadata_csv, duplicates_json, output_csv, countries,
                  years_text, training_years_text, backtest_year,
                  dominant_year, epochs, batch_size, top_n),
            daemon=True).start()

    def _fc_thread(self, metadata_csv, duplicates_json, output_csv, countries,
                   years_text, training_years_text, backtest_year,
                   dominant_year, epochs, batch_size, top_n):
        try:
            from predict_strain_growth import parse_years, run_strain_growth_forecast

            years = parse_years(years_text)
            if not years:
                raise ValueError("Enter at least one forecast year.")
            training_years = (
                parse_years(training_years_text) if training_years_text else None
            )
            if training_years is not None and not training_years:
                raise ValueError("Enter valid training years or leave the field blank.")

            self._log_to(self.fc_log, f"Metadata CSV: {metadata_csv}")
            self._log_to(self.fc_log, f"Duplicates JSON: {duplicates_json}")
            self._log_to(self.fc_log, f"Output CSV: {output_csv}")
            self._log_to(self.fc_log,
                         f"Training years: {training_years or 'all before target'}")
            self._log_to(self.fc_log, f"Dominant/backtest year: {dominant_year}")
            self.after(0, lambda: self.fc_status_var.set(
                "Training forecast model. This can take several minutes..."))

            def prog(msg, percent=None):
                first_line = msg.splitlines()[0]
                self.after(0, lambda m=first_line: self.fc_status_var.set(m))
                if percent is not None:
                    self.after(0, lambda p=percent: self.fc_progress_var.set(p))
                self._log_to(self.fc_log, msg)

            result = run_strain_growth_forecast(
                metadata_csv,
                duplicates_json,
                output_csv,
                countries=countries,
                years=years,
                training_years=training_years,
                backtest_year=backtest_year,
                dominant_year=dominant_year,
                epochs=epochs,
                batch_size=batch_size,
                top_n=top_n,
                progress_callback=prog)

            summary_lines = [
                f"Forecast rows:  {result['rows']:,}",
                f"Training rows:  {result['train_rows']:,}",
                f"Backtest rows:  {result['test_rows']:,}",
                f"Target year:    {result['target_year']}",
                f"New forecast dominant DNA changes: {result['new_forecast_dna_count']:,}",
                f"New backtest dominant DNA changes: {result['new_backtest_dna_count']:,}",
                result["r2_summary"],
                result["dominant_summary"],
                f"Saved to:       {result['output_csv']}",
            ]
            if result["dominant_backtest_csv"]:
                summary_lines.append(
                    f"Dominant CSV:   {result['dominant_backtest_csv']}"
                )
            summary_lines += ["", result["backtest_report"]]

            def _show():
                self.fc_result_text.config(state="normal")
                self.fc_result_text.delete("1.0", "end")
                self.fc_result_text.insert("end", "\n".join(summary_lines))
                self.fc_result_text.config(state="disabled")
            self.after(0, _show)

            self.after(0, lambda: self.fc_status_var.set(
                f"Done! Saved {result['rows']:,} forecast rows."))
            self.after(0, lambda: messagebox.showinfo(
                "Strain Forecast Complete",
                f"Forecast saved to:\n{result['output_csv']}\n\n"
                f"Rows: {result['rows']:,}"))
        except Exception as e:
            err = str(e)
            self._log_to(self.fc_log, f"Error: {err}")
            self.after(0, lambda: self.fc_status_var.set(f"Error: {err}"))
            self.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            self.after(0, lambda: self.fc_run_btn.config(state="normal"))

    # ────────────────────────────────────────────────────────
    #  TAB 4 — Growth Patterns
    # ────────────────────────────────────────────────────────
    def _build_growth_tab(self):
        tab = tk.Frame(self.notebook, bg="#f5f6fa")
        self.notebook.add(tab, text="  Growth Patterns  ")

        form = tk.Frame(tab, bg="#f5f6fa", padx=24, pady=14)
        form.pack(fill="both", expand=True)

        r = 0
        tk.Label(form, text="Forecast Output CSV", font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w"); r += 1
        file_row = tk.Frame(form, bg="#f5f6fa")
        file_row.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        self.gp_csv_var = tk.StringVar(value=os.path.abspath(
            "global_outbreak_forecast_2020_2025.csv"))
        tk.Entry(file_row, textvariable=self.gp_csv_var,
                 width=72, font=("Segoe UI", 10)).pack(side="left", padx=(0, 6))
        tk.Button(file_row, text="Browse…", font=("Segoe UI", 9),
                  command=self._gp_browse_csv).pack(side="left")

        opts = tk.Frame(form, bg="#f5f6fa")
        opts.grid(row=r, column=0, columnspan=2, sticky="w", pady=(0, 8)); r += 1
        self.gp_country_var = tk.StringVar(value="All Countries")
        self.gp_topn_var = tk.StringVar(value="10")
        for label, var, width in [
            ("Country Filter", self.gp_country_var, 22),
            ("Top Trends", self.gp_topn_var, 8),
        ]:
            cell = tk.Frame(opts, bg="#f5f6fa")
            cell.pack(side="left", padx=(0, 14))
            tk.Label(cell, text=label, font=("Segoe UI", 9, "bold"),
                     bg="#f5f6fa").pack(anchor="w")
            tk.Entry(cell, textvariable=var, width=width,
                     font=("Segoe UI", 10)).pack(anchor="w")

        self.gp_run_btn = tk.Button(
            form, text="Analyze Growth Patterns", font=("Segoe UI", 10, "bold"),
            bg="#2c7fb8", fg="white", padx=18, pady=5, relief="flat",
            cursor="hand2", command=self._on_gp_run)
        self.gp_run_btn.grid(row=r, column=0, sticky="w", pady=(0, 8)); r += 1

        self.gp_status_var = tk.StringVar(
            value="Load a forecast CSV to analyze strain growth patterns.")
        tk.Label(form, textvariable=self.gp_status_var,
                 font=("Segoe UI", 9), bg="#f5f6fa", fg="#555",
                 anchor="w", wraplength=850
                 ).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1

        tk.Label(form, text="Fastest Rising / Falling Strains",
                 font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w", pady=(8, 2)); r += 1
        self.gp_trend_text = tk.Text(form, height=12, font=("Consolas", 9),
                                     bg="#ecf0f1", relief="flat",
                                     state="disabled")
        self.gp_trend_text.grid(row=r, column=0, columnspan=2,
                                sticky="we", pady=(0, 8)); r += 1

        tk.Label(form, text="Dominant Predictions By Month",
                 font=("Segoe UI", 10, "bold"),
                 bg="#f5f6fa").grid(row=r, column=0, sticky="w", pady=(0, 2)); r += 1
        self.gp_dom_text = tk.Text(form, height=12, font=("Consolas", 9),
                                   bg="#ecf0f1", relief="flat",
                                   state="disabled")
        self.gp_dom_text.grid(row=r, column=0, columnspan=2,
                              sticky="we", pady=(0, 2))

    def _gp_browse_csv(self):
        p = filedialog.askopenfilename(
            title="Select forecast CSV",
            filetypes=[("CSV", "*.csv"), ("All", "*.*")])
        if p:
            self.gp_csv_var.set(p)

    def _on_gp_run(self):
        csv_path = self.gp_csv_var.get().strip()
        if not csv_path or not os.path.isfile(csv_path):
            messagebox.showwarning("No Forecast CSV",
                "Please select a valid forecast output CSV.")
            return
        try:
            top_n = int(self.gp_topn_var.get().strip())
            if top_n < 1:
                raise ValueError("Top Trends must be positive.")
        except ValueError as e:
            messagebox.showwarning("Invalid Growth Options", str(e))
            return

        country = self.gp_country_var.get().strip()
        if not country or country.lower() == "all countries":
            country = None

        self.gp_run_btn.config(state="disabled")
        threading.Thread(
            target=self._gp_thread,
            args=(csv_path, country, top_n),
            daemon=True).start()

    def _gp_thread(self, csv_path, country, top_n):
        try:
            trend_lines, dominant_lines, summary = self._analyze_growth_csv(
                csv_path, country, top_n)

            def _show():
                self.gp_trend_text.config(state="normal")
                self.gp_trend_text.delete("1.0", "end")
                self.gp_trend_text.insert("end", "\n".join(trend_lines))
                self.gp_trend_text.config(state="disabled")

                self.gp_dom_text.config(state="normal")
                self.gp_dom_text.delete("1.0", "end")
                self.gp_dom_text.insert("end", "\n".join(dominant_lines))
                self.gp_dom_text.config(state="disabled")
                self.gp_status_var.set(summary)

            self.after(0, _show)
        except Exception as e:
            err = str(e)
            self.after(0, lambda: self.gp_status_var.set(f"Error: {err}"))
            self.after(0, lambda: messagebox.showerror("Error", err))
        finally:
            self.after(0, lambda: self.gp_run_btn.config(state="normal"))

    def _analyze_growth_csv(self, csv_path, country=None, top_n=10):
        import pandas as pd

        df = pd.read_csv(csv_path, low_memory=False)
        required = {"Year", "Month", "Country", "Accession", "Probability"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                "Forecast CSV is missing required columns: " + ", ".join(sorted(missing)))

        df = df.copy()
        df["Probability"] = pd.to_numeric(df["Probability"], errors="coerce")
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df["Month"] = pd.to_numeric(df["Month"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Probability", "Year", "Month", "Country", "Accession"])
        if country:
            df = df[df["Country"].astype(str).str.casefold() == country.casefold()]
        if df.empty:
            raise ValueError("No forecast rows match the selected filter.")

        title_col = "GenBank_Title" if "GenBank_Title" in df.columns else None
        df["Time_Index"] = df["Year"].astype(int) * 12 + df["Month"].astype(int)

        trend_rows = []
        group_cols = ["Country", "Accession"]
        for (ctry, acc), group in df.sort_values("Time_Index").groupby(group_cols):
            if len(group) < 2:
                continue
            first = group.iloc[0]
            last = group.iloc[-1]
            peak = group.loc[group["Probability"].idxmax()]
            delta = float(last["Probability"] - first["Probability"])
            title = ""
            if title_col:
                title = str(last.get(title_col) or first.get(title_col) or "")
            trend_rows.append({
                "country": ctry,
                "accession": acc,
                "title": title[:72],
                "start": float(first["Probability"]),
                "end": float(last["Probability"]),
                "delta": delta,
                "peak": float(peak["Probability"]),
                "peak_label": f"{int(peak['Year'])}-{int(peak['Month']):02d}",
            })

        if not trend_rows:
            raise ValueError("Need at least two months per strain to compute growth.")

        rising = sorted(trend_rows, key=lambda row: row["delta"], reverse=True)[:top_n]
        falling = sorted(trend_rows, key=lambda row: row["delta"])[:top_n]

        trend_lines = [
            f"Rows analyzed: {len(df):,}",
            f"Country filter: {country or 'All Countries'}",
            "",
            "FASTEST RISING STRAINS",
            "Country              Accession           Start    End      Delta    Peak",
            "-" * 82,
        ]
        trend_lines += [self._format_growth_row(row) for row in rising]
        trend_lines += [
            "",
            "FASTEST FALLING STRAINS",
            "Country              Accession           Start    End      Delta    Peak",
            "-" * 82,
        ]
        trend_lines += [self._format_growth_row(row) for row in falling]

        dominant = (
            df.sort_values("Probability", ascending=False)
            .drop_duplicates(["Country", "Year", "Month"])
            .sort_values(["Country", "Year", "Month"])
        )
        dominant_lines = [
            "Country              Date      Accession           Probability  Title",
            "-" * 100,
        ]
        for _, row in dominant.head(250).iterrows():
            title = str(row.get(title_col, ""))[:42] if title_col else ""
            dominant_lines.append(
                f"{str(row['Country'])[:20]:20}  "
                f"{int(row['Year'])}-{int(row['Month']):02d}  "
                f"{str(row['Accession'])[:18]:18}  "
                f"{float(row['Probability']):10.4f}  "
                f"{title}"
            )
        if len(dominant) > 250:
            dominant_lines.append(f"... {len(dominant) - 250:,} more rows not shown")

        summary = (
            f"Analyzed {len(df):,} forecast rows across "
            f"{df['Country'].nunique():,} countries and {df['Accession'].nunique():,} strains."
        )
        return trend_lines, dominant_lines, summary

    def _format_growth_row(self, row):
        return (
            f"{str(row['country'])[:20]:20}  "
            f"{str(row['accession'])[:18]:18}  "
            f"{row['start']:7.4f}  "
            f"{row['end']:7.4f}  "
            f"{row['delta']:+8.4f}  "
            f"{row['peak']:.4f}@{row['peak_label']}  "
            f"{row['title']}"
        )


# ──────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
