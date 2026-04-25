"""
Tkinter GUI for the NCBI Virus Sequence Downloader.

Run this file when you want the desktop interface. The download and
extraction logic lives in ncbi_virus_downloader.py so the CLI and GUI
share the same behavior.
"""

import os
import platform
import subprocess
import tempfile
import threading
import zipfile
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ncbi_virus_downloader import (
    LOCATIONS,
    VIRUS_TYPES,
    _app_dir,
    _date_choices,
    _resolve_date,
    ensure_cli,
    extract_csv,
    extract_fasta,
    run_datasets_download,
)


class NCBIVirusDownloader(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("NCBI Virus Sequence Downloader")
        self.geometry("720x620")
        self.resizable(False, False)
        self.configure(bg="#f5f6fa")

        self._cli_paths = None
        self._last_zip = None
        self._build_ui()

        threading.Thread(target=self._ensure_cli_thread, daemon=True).start()

    def _build_ui(self):
        header = tk.Frame(self, bg="#2c3e50", height=60)
        header.pack(fill="x")
        header.pack_propagate(False)
        tk.Label(
            header,
            text="NCBI Virus Sequence Downloader",
            font=("Segoe UI", 16, "bold"),
            fg="white",
            bg="#2c3e50",
        ).pack(pady=14)

        main = tk.Frame(self, bg="#f5f6fa", padx=30, pady=20)
        main.pack(fill="both", expand=True)

        tk.Label(
            main,
            text="Virus Type",
            font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa",
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))
        self.virus_var = tk.StringVar(value=list(VIRUS_TYPES.keys())[0])
        ttk.Combobox(
            main,
            textvariable=self.virus_var,
            values=list(VIRUS_TYPES.keys()),
            state="readonly",
            width=48,
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        tk.Label(
            main,
            text="Location",
            font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa",
            anchor="w",
        ).grid(row=2, column=0, sticky="w", pady=(0, 2))
        self.location_var = tk.StringVar(value="Any Location")
        ttk.Combobox(
            main,
            textvariable=self.location_var,
            values=LOCATIONS,
            state="readonly",
            width=48,
            font=("Segoe UI", 10),
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(0, 12))

        tk.Label(
            main,
            text="Released After",
            font=("Segoe UI", 10, "bold"),
            bg="#f5f6fa",
            anchor="w",
        ).grid(row=4, column=0, sticky="w", pady=(0, 2))
        self.date_var = tk.StringVar(value="Any Date")
        ttk.Combobox(
            main,
            textvariable=self.date_var,
            values=_date_choices(),
            state="readonly",
            width=48,
            font=("Segoe UI", 10),
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(0, 12))

        self.complete_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            main,
            text="Complete sequences only",
            variable=self.complete_var,
            font=("Segoe UI", 10),
            bg="#f5f6fa",
        ).grid(row=6, column=0, sticky="w", pady=(0, 12))

        btn_frame = tk.Frame(main, bg="#f5f6fa")
        btn_frame.grid(row=7, column=0, columnspan=2, sticky="w", pady=(4, 8))

        self.fasta_btn = tk.Button(
            btn_frame,
            text="Download FASTA",
            font=("Segoe UI", 10, "bold"),
            bg="#27ae60",
            fg="white",
            activebackground="#219a52",
            padx=20,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=lambda: self._on_download("fasta"),
        )
        self.fasta_btn.pack(side="left", padx=(0, 10))

        self.csv_btn = tk.Button(
            btn_frame,
            text="Download CSV",
            font=("Segoe UI", 10, "bold"),
            bg="#e67e22",
            fg="white",
            activebackground="#d35400",
            padx=20,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=lambda: self._on_download("csv"),
        )
        self.csv_btn.pack(side="left", padx=(0, 10))

        self.both_btn = tk.Button(
            btn_frame,
            text="Download Both",
            font=("Segoe UI", 10, "bold"),
            bg="#3498db",
            fg="white",
            activebackground="#2980b9",
            padx=20,
            pady=6,
            relief="flat",
            cursor="hand2",
            command=lambda: self._on_download("both"),
        )
        self.both_btn.pack(side="left")

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            main, variable=self.progress_var, maximum=100, length=660, mode="indeterminate"
        )
        self.progress_bar.grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 4))

        self.status_var = tk.StringVar(value="Checking for NCBI Datasets CLI...")
        tk.Label(
            main,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg="#f5f6fa",
            fg="#555",
            anchor="w",
            wraplength=660,
        ).grid(row=9, column=0, columnspan=2, sticky="w")

        tk.Label(
            main,
            text="Log",
            font=("Segoe UI", 9, "bold"),
            bg="#f5f6fa",
            anchor="w",
        ).grid(row=10, column=0, sticky="w", pady=(8, 2))
        self.log_text = tk.Text(
            main,
            height=7,
            width=82,
            font=("Consolas", 9),
            bg="#ecf0f1",
            relief="flat",
            state="disabled",
        )
        self.log_text.grid(row=11, column=0, columnspan=2, sticky="we")

        self._set_buttons_enabled(False)

    def _log(self, msg):
        def _do():
            self.log_text.config(state="normal")
            self.log_text.insert("end", f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
            self.log_text.see("end")
            self.log_text.config(state="disabled")

        self.after(0, _do)

    def _set_status(self, msg):
        self.after(0, lambda: self.status_var.set(msg))

    def _set_buttons_enabled(self, enabled):
        state = "normal" if enabled else "disabled"
        for btn in (self.fasta_btn, self.csv_btn, self.both_btn):
            btn.config(state=state)

    def _ensure_cli_thread(self):
        try:
            self._log(f"Platform: {platform.system()} {platform.machine()}")
            self._log(f"App directory: {_app_dir()}")

            def on_progress(msg):
                self._set_status(msg)
                self._log(msg)

            self._cli_paths = ensure_cli(progress_callback=on_progress)
            self._log(f"datasets binary: {self._cli_paths['datasets']}")
            self._log(f"dataformat binary: {self._cli_paths['dataformat']}")

            result = subprocess.run(
                [self._cli_paths["datasets"], "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            self._log(f"CLI version: {result.stdout.strip()}")

            self._set_status("Ready. Select options and click a Download button.")
            self.after(0, lambda: self._set_buttons_enabled(True))
        except Exception as e:
            self._log(f"CLI setup error: {e}")
            self._set_status(f"Error setting up CLI: {e}")

    def _on_download(self, fmt):
        if not self._cli_paths:
            messagebox.showwarning(
                "Not Ready", "NCBI Datasets CLI is still being set up. Please wait."
            )
            return

        virus_label = self.virus_var.get().replace(" ", "_").replace("–", "")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"ncbi_{virus_label}_{stamp}"

        if fmt == "fasta":
            path = filedialog.asksaveasfilename(
                title="Save FASTA file",
                defaultextension=".fasta",
                initialfile=base_name + ".fasta",
                filetypes=[("FASTA files", "*.fasta *.fna"), ("All files", "*.*")],
            )
            if not path:
                return
            paths = {"fasta": path}
        elif fmt == "csv":
            path = filedialog.asksaveasfilename(
                title="Save CSV file",
                defaultextension=".csv",
                initialfile=base_name + ".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            paths = {"csv": path}
        else:
            folder = filedialog.askdirectory(title="Choose folder for FASTA + CSV")
            if not folder:
                return
            paths = {
                "fasta": os.path.join(folder, base_name + ".fasta"),
                "csv": os.path.join(folder, base_name + ".csv"),
            }

        self._set_buttons_enabled(False)
        self.progress_bar.start(15)
        threading.Thread(target=self._download_thread, args=(paths,), daemon=True).start()

    def _download_thread(self, paths):
        tmp_zip = None
        try:
            virus = self.virus_var.get()
            taxon = VIRUS_TYPES[virus]
            location = self.location_var.get()
            released_after = _resolve_date(self.date_var.get())
            complete_only = self.complete_var.get()

            self._log(f"Virus: {virus} (taxon: {taxon})")
            self._log(f"Location: {location} | Released after: {released_after or 'any'}")
            self._set_status("Downloading from NCBI (this may take a moment)...")

            tmp_zip = os.path.join(tempfile.gettempdir(), f"ncbi_download_{os.getpid()}.zip")
            run_datasets_download(
                self._cli_paths,
                taxon,
                location,
                released_after,
                tmp_zip,
                complete_only=complete_only,
                log_callback=lambda m: self._log(m),
            )

            with zipfile.ZipFile(tmp_zip) as z:
                self._log(f"Zip contents: {z.namelist()}")

            if "fasta" in paths:
                self._set_status("Extracting FASTA...")
                extract_fasta(tmp_zip, paths["fasta"])
                size_kb = os.path.getsize(paths["fasta"]) / 1024
                self._log(f"FASTA saved: {paths['fasta']} ({size_kb:,.0f} KB)")

            if "csv" in paths:
                self._set_status("Converting metadata to CSV...")
                extract_csv(self._cli_paths, tmp_zip, paths["csv"])
                size_kb = os.path.getsize(paths["csv"]) / 1024
                self._log(f"CSV saved: {paths['csv']} ({size_kb:,.0f} KB)")

            saved = " and ".join(os.path.basename(p) for p in paths.values())
            self._set_status(f"Done! Saved: {saved}")
            self.after(
                0,
                lambda: messagebox.showinfo(
                    "Download Complete", "Files saved:\n" + "\n".join(paths.values())
                ),
            )
        except FileNotFoundError as e:
            self._log(f"No data: {e}")
            self._set_status("No sequences found - try broader filters.")
            self.after(0, lambda: messagebox.showwarning("No Sequences Found", str(e)))
        except Exception as e:
            self._log(f"Error: {e}")
            self._set_status(f"Download failed: {e}")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))
        finally:
            if tmp_zip and os.path.isfile(tmp_zip):
                try:
                    os.remove(tmp_zip)
                except OSError:
                    pass
            self.after(0, lambda: self.progress_bar.stop())
            self.after(0, lambda: self._set_buttons_enabled(True))


def main():
    app = NCBIVirusDownloader()
    app.mainloop()


if __name__ == "__main__":
    main()
