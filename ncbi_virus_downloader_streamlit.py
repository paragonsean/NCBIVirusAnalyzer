"""
Streamlit version of NCBI Virus Tools.

Run with:
    streamlit run ncbi_virus_downloader_streamlit.py
"""

from datetime import datetime
import os
import tempfile
import zipfile

import pandas as pd
import streamlit as st

from ncbi_virus_downloader import (
    LOCATIONS,
    VIRUS_TYPES,
    _date_choices,
    _resolve_date,
    build_duplicate_dict,
    ensure_cli,
    extract_csv,
    extract_fasta,
    run_datasets_download,
)
from predict_strain_growth import parse_years, run_strain_growth_forecast


def analyze_growth_csv(csv_path, country=None, top_n=10):
    df = pd.read_csv(csv_path, low_memory=False)
    required = {"Year", "Month", "Country", "Accession", "Probability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            "Forecast CSV is missing required columns: " + ", ".join(sorted(missing))
        )

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
        trend_rows.append(
            {
                "Country": ctry,
                "Accession": acc,
                "Start_Prob": float(first["Probability"]),
                "End_Prob": float(last["Probability"]),
                "Delta": delta,
                "Peak_Prob": float(peak["Probability"]),
                "Peak_Label": f"{int(peak['Year'])}-{int(peak['Month']):02d}",
                "Title": title[:120],
            }
        )

    if not trend_rows:
        raise ValueError("Need at least two months per strain to compute growth.")

    trend_df = pd.DataFrame(trend_rows)
    rising_df = trend_df.sort_values("Delta", ascending=False).head(top_n)
    falling_df = trend_df.sort_values("Delta", ascending=True).head(top_n)

    dominant_df = (
        df.sort_values("Probability", ascending=False)
        .drop_duplicates(["Country", "Year", "Month"])
        .sort_values(["Country", "Year", "Month"])
    )
    if title_col is None:
        dominant_df["GenBank_Title"] = ""

    summary = (
        f"Analyzed {len(df):,} rows across "
        f"{df['Country'].nunique():,} countries and {df['Accession'].nunique():,} strains."
    )
    return rising_df, falling_df, dominant_df, summary


def tab_download():
    st.subheader("Download")
    st.caption("Fetch FASTA and/or CSV from NCBI Datasets CLI.")

    col1, col2 = st.columns(2)
    with col1:
        virus = st.selectbox("Virus Type", list(VIRUS_TYPES.keys()))
        location = st.selectbox("Location", LOCATIONS, index=0)
    with col2:
        date_choice = st.selectbox("Released After", _date_choices(), index=0)
        complete_only = st.checkbox("Complete sequences only", value=False)

    fmt = st.selectbox("Format", ["fasta", "csv", "both"])
    output_dir = st.text_input("Output directory", value=os.getcwd())

    if st.button("Run Download", type="primary"):
        try:
            os.makedirs(output_dir, exist_ok=True)
            cli_paths = ensure_cli(progress_callback=lambda m: st.write(m))
            taxon = VIRUS_TYPES[virus]
            released_after = _resolve_date(date_choice)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = f"ncbi_{virus.replace(' ', '_').replace('–', '')}_{stamp}"
            tmp_zip = os.path.join(tempfile.gettempdir(), f"ncbi_dl_{os.getpid()}.zip")

            run_datasets_download(
                cli_paths,
                taxon,
                location,
                released_after,
                tmp_zip,
                complete_only=complete_only,
                log_callback=lambda m: st.write(m),
            )
            with zipfile.ZipFile(tmp_zip) as z:
                st.write("Zip contents:", z.namelist())

            saved_paths = []
            if fmt in ("fasta", "both"):
                fasta_path = os.path.join(output_dir, base + ".fasta")
                extract_fasta(tmp_zip, fasta_path)
                saved_paths.append(fasta_path)
            if fmt in ("csv", "both"):
                csv_path = os.path.join(output_dir, base + ".csv")
                extract_csv(cli_paths, tmp_zip, csv_path)
                saved_paths.append(csv_path)
            st.success("Download complete.")
            st.write("Saved files:")
            for p in saved_paths:
                st.code(p)
        except Exception as exc:
            st.error(str(exc))
        finally:
            if "tmp_zip" in locals() and os.path.isfile(tmp_zip):
                try:
                    os.remove(tmp_zip)
                except OSError:
                    pass


def tab_pattern_machine():
    st.subheader("Pattern Machine")
    st.caption("Build duplicate-sequence JSON from FASTA.")

    fasta_path = st.text_input("FASTA input path")
    output_json = st.text_input("Output duplicates JSON path")
    metadata_csv = st.text_input("Metadata CSV (optional)")
    embed_metadata = st.checkbox(
        "Embed metadata in duplicates JSON (larger file; not needed for training)", value=False
    )
    max_seqs_text = st.text_input("Max sequences (optional)")

    if st.button("Find Duplicates", type="primary"):
        try:
            if not fasta_path or not os.path.isfile(fasta_path):
                raise ValueError("Provide a valid FASTA input path.")
            if not output_json:
                raise ValueError("Provide an output JSON path.")
            if embed_metadata and (not metadata_csv or not os.path.isfile(metadata_csv)):
                raise ValueError("Provide a valid metadata CSV or disable embedding.")

            max_seqs = int(max_seqs_text) if max_seqs_text.strip().isdigit() else None
            result = build_duplicate_dict(
                fasta_path,
                output_json,
                metadata_csv=metadata_csv if embed_metadata else None,
                max_seqs=max_seqs,
                progress_callback=lambda m: st.write(m),
            )
            st.success(f"Saved: {output_json}")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))


def tab_strain_forecast():
    st.subheader("Strain Forecast")
    st.caption("Train and generate forecast + dominant backtest CSVs.")

    metadata_csv = st.text_input("Metadata CSV path")
    duplicates_json = st.text_input("Duplicates JSON path")
    output_csv = st.text_input(
        "Output forecast CSV path",
        value=os.path.abspath("global_outbreak_forecast_2020_2025.csv"),
    )
    countries_text = st.text_input(
        "Countries (comma-separated)",
        value="USA, Canada, France, Mexico, United Kingdom",
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        forecast_years_text = st.text_input("Forecast Years", value="2020-2025")
        training_years_text = st.text_input("Training Years", value="2020-2024")
    with c2:
        backtest_year = st.number_input("Backtest Year", min_value=1900, max_value=2100, value=2025)
        dominant_year = st.number_input("Dominant Year", min_value=1900, max_value=2100, value=2025)
    with c3:
        epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=10)
        batch_size = st.number_input("Batch Size", min_value=1, max_value=8192, value=256)
    with c4:
        top_n = st.number_input("Top N", min_value=1, max_value=100, value=3)

    if st.button("Run Forecast", type="primary"):
        try:
            if not metadata_csv or not os.path.isfile(metadata_csv):
                raise ValueError("Provide a valid metadata CSV path.")
            if not duplicates_json or not os.path.isfile(duplicates_json):
                raise ValueError("Provide a valid duplicates JSON path.")
            if not output_csv:
                raise ValueError("Provide an output forecast CSV path.")

            countries = [c.strip() for c in countries_text.split(",") if c.strip()]
            if not countries:
                raise ValueError("Enter at least one country.")
            years = parse_years(forecast_years_text)
            if not years:
                raise ValueError("Enter at least one forecast year.")
            training_years = parse_years(training_years_text) if training_years_text.strip() else None

            progress = st.progress(0)
            status = st.empty()

            def prog(msg, percent=None):
                status.write(msg)
                if percent is not None:
                    progress.progress(int(max(0, min(100, percent))))

            result = run_strain_growth_forecast(
                metadata_csv=metadata_csv,
                duplicates_json=duplicates_json,
                output_csv=output_csv,
                countries=countries,
                years=years,
                training_years=training_years,
                backtest_year=int(backtest_year),
                dominant_year=int(dominant_year),
                epochs=int(epochs),
                batch_size=int(batch_size),
                top_n=int(top_n),
                progress_callback=prog,
            )
            progress.progress(100)
            st.success("Forecast complete.")
            st.json(result)
        except Exception as exc:
            st.error(str(exc))


def tab_growth_patterns():
    st.subheader("Growth Patterns")
    st.caption("Analyze forecast CSV trends and dominant predictions.")

    csv_path = st.text_input(
        "Forecast CSV path",
        value=os.path.abspath("global_outbreak_forecast_2020_2025.csv"),
    )
    col1, col2 = st.columns(2)
    with col1:
        country_filter = st.text_input("Country filter (optional)")
    with col2:
        top_n = st.number_input("Top trends", min_value=1, max_value=200, value=10)

    if st.button("Analyze Growth Patterns", type="primary"):
        try:
            if not csv_path or not os.path.isfile(csv_path):
                raise ValueError("Provide a valid forecast CSV path.")
            rising_df, falling_df, dominant_df, summary = analyze_growth_csv(
                csv_path,
                country=country_filter.strip() or None,
                top_n=int(top_n),
            )
            st.success(summary)
            st.markdown("**Fastest Rising Strains**")
            st.dataframe(rising_df, use_container_width=True, height=260)
            st.markdown("**Fastest Falling Strains**")
            st.dataframe(falling_df, use_container_width=True, height=260)
            st.markdown("**Dominant Predictions by Month**")
            st.dataframe(dominant_df.head(500), use_container_width=True, height=360)
        except Exception as exc:
            st.error(str(exc))


def main():
    st.set_page_config(page_title="NCBI Virus Tools (Streamlit)", layout="wide")
    st.title("NCBI Virus Tools (Streamlit)")
    st.caption("Streamlit port of downloader, pattern machine, forecast, and growth analysis.")

    tabs = st.tabs(["Download", "Pattern Machine", "Strain Forecast", "Growth Patterns"])
    with tabs[0]:
        tab_download()
    with tabs[1]:
        tab_pattern_machine()
    with tabs[2]:
        tab_strain_forecast()
    with tabs[3]:
        tab_growth_patterns()


if __name__ == "__main__":
    main()
