import csv
import os
import subprocess
import zipfile

import pytest

import ncbi_virus_downloader as mod


@pytest.mark.network
def test_download_binary_live_network(monkeypatch, tmp_path):
    if os.environ.get("RUN_NETWORK_TESTS") != "1":
        pytest.skip("Set RUN_NETWORK_TESTS=1 to run live network download test.")

    monkeypatch.setattr(mod, "_app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        mod,
        "_download_url_for",
        lambda _idx: "https://ftp.ncbi.nlm.nih.gov/README.ftp",
    )

    progress_messages = []
    out_path = mod._download_binary("datasets", 0, progress_callback=progress_messages.append)

    assert os.path.isfile(out_path)
    assert os.path.getsize(out_path) > 0
    with open(out_path, "rb") as fh:
        assert fh.read(16).strip() != b""
    assert any("Downloading datasets" in msg for msg in progress_messages)


def test_date_helpers():
    choices = mod._date_choices()
    assert "Any Date" in choices
    assert mod._resolve_date("Any Date") is None
    assert mod._resolve_date("Last 30 days") is not None


def test_app_dir_and_binary_name(monkeypatch):
    monkeypatch.setattr(mod.sys, "frozen", False, raising=False)
    app_dir = mod._app_dir()
    assert os.path.isdir(app_dir)
    assert mod._binary_name("datasets").startswith("datasets")


def test_find_binary_prefers_local(monkeypatch, tmp_path):
    local = tmp_path / "datasets.exe"
    local.write_text("x", encoding="utf-8")
    monkeypatch.setattr(mod, "_app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    found = mod._find_binary("datasets")
    assert found == str(local)


def test_find_binary_falls_back_to_which(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mod.shutil, "which", lambda _: "C:/tools/datasets.exe")
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    found = mod._find_binary("datasets")
    assert found == "C:/tools/datasets.exe"


def test_download_url_for_valid_and_invalid(monkeypatch):
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(mod.platform, "machine", lambda: "AMD64")
    url = mod._download_url_for(0)
    assert url.endswith("/datasets.exe")

    monkeypatch.setattr(mod.platform, "system", lambda: "UnknownOS")
    monkeypatch.setattr(mod.platform, "machine", lambda: "x86_64")
    with pytest.raises(RuntimeError):
        mod._download_url_for(0)


def test_download_binary(monkeypatch, tmp_path):
    monkeypatch.setattr(mod, "_app_dir", lambda: str(tmp_path))
    monkeypatch.setattr(mod.platform, "system", lambda: "Windows")
    monkeypatch.setattr(mod, "_download_url_for", lambda _: "https://example.test/tool")

    class FakeResp:
        def __init__(self):
            self.headers = {"Content-Length": "6"}
            self._parts = [b"abc", b"def", b""]

        def read(self, _):
            return self._parts.pop(0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResp())

    logs = []
    out = mod._download_binary("datasets", 0, progress_callback=logs.append)
    assert os.path.isfile(out)
    assert os.path.basename(out) == "datasets.exe"
    assert any("Downloading datasets" in m for m in logs)


def test_ensure_cli_downloads_missing(monkeypatch):
    calls = []

    def fake_find(tool):
        return None if tool == "datasets" else "/ok/dataformat.exe"

    def fake_download(tool, idx, _cb):
        calls.append((tool, idx))
        return f"/tmp/{tool}"

    monkeypatch.setattr(mod, "_find_binary", fake_find)
    monkeypatch.setattr(mod, "_download_binary", fake_download)
    out = mod.ensure_cli(progress_callback=lambda _m: None)
    assert out["datasets"] == "/tmp/datasets"
    assert out["dataformat"] == "/ok/dataformat.exe"
    assert calls == [("datasets", 0)]


def test_run_datasets_download_success(monkeypatch, tmp_path):
    out_zip = tmp_path / "out.zip"
    out_zip.write_bytes(b"ok")

    class FakeProc:
        def __init__(self):
            self.stdout = iter(["downloading 45%\n", "done 100%\n"])
            self.returncode = 0

        def wait(self, timeout=None):
            assert timeout == 600
            return 0

        def kill(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    logs = []
    progress = []
    result = mod.run_datasets_download(
        {"datasets": "datasets.exe"},
        "11320",
        "USA",
        "2024-01-01",
        str(out_zip),
        complete_only=True,
        log_callback=logs.append,
        progress_callback=lambda p, s: progress.append((p, s)),
    )
    assert result == str(out_zip)
    assert any("Running:" in m for m in logs)
    assert any(p == 100 for p, _ in progress)


def test_run_datasets_download_error(monkeypatch, tmp_path):
    out_zip = tmp_path / "out.zip"

    class FakeProc:
        def __init__(self):
            self.stdout = iter(["fatal error\n"])
            self.returncode = 1

        def wait(self, timeout=None):
            return 1

        def kill(self):
            return None

    monkeypatch.setattr(mod.subprocess, "Popen", lambda *args, **kwargs: FakeProc())
    with pytest.raises(RuntimeError):
        mod.run_datasets_download(
            {"datasets": "datasets.exe"},
            "11320",
            "Any Location",
            None,
            str(out_zip),
        )


def test_download_permutations_for_virus_location_date_and_complete_only(monkeypatch, tmp_path):
    out_zip = tmp_path / "perm_out.zip"
    out_zip.write_bytes(b"ok")
    captured_cmds = []

    class FakeProc:
        def __init__(self):
            self.stdout = iter(["progress 100%\n"])
            self.returncode = 0

        def wait(self, timeout=None):
            assert timeout == 600
            return 0

        def kill(self):
            return None

    def fake_popen(cmd, **kwargs):
        captured_cmds.append(list(cmd))
        return FakeProc()

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    total = 0
    for taxon in mod.VIRUS_TYPES.values():
        for location in mod.LOCATIONS:
            for date_label in mod._date_choices():
                released_after = mod._resolve_date(date_label)
                for complete_only in (False, True):
                    total += 1
                    mod.run_datasets_download(
                        {"datasets": "datasets.exe"},
                        taxon,
                        location,
                        released_after,
                        str(out_zip),
                        complete_only=complete_only,
                    )
                    cmd = captured_cmds[-1]
                    assert cmd[:6] == [
                        "datasets.exe",
                        "download",
                        "virus",
                        "genome",
                        "taxon",
                        taxon,
                    ]
                    assert "--filename" in cmd
                    assert str(out_zip) in cmd

                    if location == "Any Location":
                        assert "--geo-location" not in cmd
                    else:
                        i = cmd.index("--geo-location")
                        assert cmd[i + 1] == location

                    if released_after is None:
                        assert "--released-after" not in cmd
                    else:
                        i = cmd.index("--released-after")
                        assert cmd[i + 1] == released_after

                    if complete_only:
                        assert "--complete-only" in cmd
                    else:
                        assert "--complete-only" not in cmd

    expected_total = (
        len(mod.VIRUS_TYPES)
        * len(mod.LOCATIONS)
        * len(mod._date_choices())
        * 2
    )
    assert total == expected_total
    assert len(captured_cmds) == expected_total


def test_extract_fasta(tmp_path):
    zip_path = tmp_path / "in.zip"
    dest = tmp_path / "genomic.fna"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("ncbi_dataset/data/genomic.fna", ">x\nACGT\n")
    mod.extract_fasta(str(zip_path), str(dest))
    assert dest.read_text(encoding="utf-8").startswith(">x")


def test_extract_csv(monkeypatch, tmp_path):
    zip_path = tmp_path / "in.zip"
    out_csv = tmp_path / "out.csv"
    with zipfile.ZipFile(zip_path, "w") as z:
        z.writestr("ncbi_dataset/data/data_report.jsonl", "{}\n")

    class Res:
        returncode = 0
        stdout = "a\tb\n1\t2\n"
        stderr = ""

    monkeypatch.setattr(mod.subprocess, "run", lambda *args, **kwargs: Res())
    mod.extract_csv({"dataformat": "dataformat.exe"}, str(zip_path), str(out_csv))
    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert rows[0] == ["a", "b"]
    assert rows[1] == ["1", "2"]


def test_sequence_helpers_and_parse(tmp_path):
    assert mod._clean_dna("Acgtnx-") == "ACGT"
    assert mod._extract_fasta_id(">abc desc") == "abc"

    fasta = tmp_path / "in.fasta"
    fasta.write_text(">a\nACTGNN\n>b foo\nacgt\n", encoding="utf-8")
    entries = list(mod.parse_fasta_entries(str(fasta)))
    assert entries == [("a", "ACTG"), ("b", "ACGT")]


def test_clean_csv_value_item_object():
    class Box:
        def item(self):
            return 7

    assert mod._clean_csv_value(Box()) == 7


def test_load_metadata_by_accession(tmp_path):
    pd = pytest.importorskip("pandas")
    csv_path = tmp_path / "meta.csv"
    pd.DataFrame(
        [
            {"Accession": "A1", "Host": "Human"},
            {"Accession": "A2", "Host": "Bird"},
        ]
    ).to_csv(csv_path, index=False)
    out = mod.load_metadata_by_accession(str(csv_path))
    assert out["A1"]["Host"] == "Human"
    assert out["A2"]["Host"] == "Bird"


def test_build_duplicate_dict(tmp_path):
    fasta = tmp_path / "in.fasta"
    out_json = tmp_path / "out.json"
    fasta.write_text(">a\nACGT\n>b\nACGT\n>c\nTTTT\n", encoding="utf-8")
    result = mod.build_duplicate_dict(str(fasta), str(out_json))
    assert result["total_sequences"] == 3
    assert result["unique_sequences"] == 2
    assert result["duplicate_sequences"] == 1
    assert out_json.is_file()

