# Viral Forecasting Program User and Technical Documentation Draft

## Wildfire Earthdata ML Pipeline

This branch adds a North America wildfire analysis pipeline alongside the
original NCBI virus tools. The wildfire workflow is designed to test whether
groundwater/root-zone moisture and atmospheric aridity signals can help predict
monthly regional fire activity and then backtest those predictions against the
latest available 2026 fire season observations.

### Data Sources

- Fire target: NASA FIRMS archive CSV for active fire detections. The pipeline
  reads latitude, longitude, acquisition date, brightness, and confidence, then
  aggregates points to monthly region-level `fire_count` targets.
- Optional burned-area target: NASA MCD64A1 monthly burned area. The helper in
  `wildfire/ingest/mcd64a1.py` aggregates extracted burned-area pixels by
  month-region.
- Groundwater predictors: NASA GES DISC `GRACEDADM_CLSM025GL_7D`, using
  `gws_inst` and `rtzsm_inst`.
- Atmospheric aridity predictors: Copernicus ERA5 monthly single-level
  temperature and dewpoint. `wildfire/ingest/era5.py` computes vapor pressure
  deficit (VPD) from `2m_temperature` and `2m_dewpoint_temperature`.

### Setup

Install the wildfire-specific dependencies:

```powershell
python -m pip install -r requirements-wildfire.txt
```

For full data acquisition, configure:

- NASA Earthdata Login for `earthaccess`.
- Copernicus CDS API credentials for `cdsapi`.
- A FIRMS archive CSV downloaded for North America from 2015-01-01 through the
  latest available date.

The unit tests do not require live credentials or network access.

### Streamlit Downloader App

Launch the wildfire download UI from the `wildfire_ml` environment:

```powershell
C:\Users\spoca\anaconda3\envs\wildfire_ml\python.exe -m streamlit run wildfire_downloader_streamlit.py
```

The app includes tabs for:

- FIRMS active-fire CSV download through the FIRMS Area API.
- GRACE `GRACEDADM_CLSM025GL_7D` download through NASA Earthdata.
- ERA5 temperature/dewpoint NetCDF download through CDS.
- MCD64A1 monthly burned-area granule download through NASA Earthdata.
- Feature-table generation from downloaded files.
- Baseline 2026 backtesting.

### Build Features

Start with a local FIRMS CSV and optional GRACE/ERA5 NetCDF files:

```powershell
python wildfire_pipeline.py build-features `
  --firms-csv data/raw/firms_north_america_2015_2026.csv `
  --grace-nc data/raw/grace/*.nc `
  --era5-nc data/raw/era5/*.nc `
  --output-csv data/processed/wildfire_monthly_training.csv
```

The output table includes monthly targets, predictor summaries, 1-3 month lags,
rolling drought indicators, and a `severe_fire_month` label.

### Backtest 2026

```powershell
python wildfire_pipeline.py backtest `
  --feature-csv data/processed/wildfire_monthly_training.csv `
  --output-dir wildfire_output `
  --target-column fire_count `
  --backtest-year 2026 `
  --classifier
```

The regression backtest writes prediction CSVs and metrics JSON with MAE, RMSE,
R2, and top-risk recall. The optional classifier writes severe-fire-month
probabilities and classification metrics.

### Important Limitations

- 2026 backtests are only as complete as the latest available FIRMS, GRACE, and
  ERA5 data. GRACE archive products can have latency.
- FIRMS active-fire counts and MCD64A1 burned area answer related but different
  questions. Use FIRMS first for quick iteration, then compare with MCD64A1 for
  burned-area modeling.
- The first model is intentionally a tabular baseline. It is easier to validate
  and inspect than the original virus CNN and is a better starting point for
  climate/hydrology features.

Version: Draft prepared 2026-04-24 23:00

Source files analyzed:
- `C:\Users\spoca\Downloads\Plants\ncbi_virus_downloader.py`
- `C:\Users\spoca\Downloads\Plants\predict_strain_growth.py`
- `C:\Users\spoca\Downloads\Plants\global_outbreak_forecast_2020_2025new.csv`

Supplied image placeholders included for:
- `C:\Users\spoca\Desktop\download_1.png`
- `C:\Users\spoca\Desktop\download_two.png`
- `C:\Users\spoca\Desktop\pattern_machine.png`
- `C:\Users\spoca\Desktop\strain_forecast.png`
- `C:\Users\spoca\Desktop\analyze_growth_patterns.png`

---

## 1. Overview
Youtube Videos of usage: https://youtu.be/qRCJB6b0bh4 https://youtu.be/-b5xst1zXWI
This document is a combined user manual and technical architecture draft for the viral forecasting program implemented across two Python source files: `ncbi_virus_downloader.py` and `predict_strain_growth.py`. Together, these files define a desktop workflow that starts with sequence acquisition from NCBI, moves through DNA sequence deduplication and metadata alignment, trains a neural network to estimate which strains are likely to become prominent, and then summarizes the forecast in a post-analysis dashboard.

At the user level, the application presents a single Tkinter desktop window titled **NCBI Virus Tools**. That window is organized into four tabs:

1. Download
2. Pattern Machine
3. Strain Forecast
4. Growth Patterns

The software is designed to lower the barrier to viral sequence analysis. Instead of forcing a user to manually install and operate the NCBI Datasets command-line tools, parse multi-file archive structures, write duplicate-collapsing code, normalize metadata columns, and manually train a machine learning model, the program assembles those steps into a guided interface. The result is a workflow that can be understood as an end-to-end forecasting pipeline:

- Acquire viral genomes and metadata from NCBI.
- Extract sequence and report files into analyst-friendly formats.
- Collapse repeated identical DNA sequences into master sequence groups.
- Link those groups back to collection metadata such as country and date.
- Compute a training label that represents whether a strain becomes epidemiologically important in the following month.
- Train a late-fusion neural network on DNA plus metadata.
- Backtest the model on a held-out year.
- Score all known strains for many countries and months.
- Analyze rising, falling, and dominant predicted strains.

Technically, the system is best understood as a hybrid of three subsystems:

- A data acquisition subsystem that wraps the NCBI Datasets CLI.
- A sequence normalization and deduplication subsystem that creates the model-ready JSON input.
- A machine learning forecasting subsystem that uses one-hot encoded viral DNA together with geographic and temporal metadata.

The supplied forecast CSV, `global_outbreak_forecast_2020_2025new.csv`, shows the shape of the downstream product. It contains columns for `Year`, `Month`, `Country`, `Accession`, `Probability`, and `GenBank_Title`, which means the final forecast output is a month-level ranking of likely strains by geography. In the sample rows, France appears for early 2020 with several H3N2 accessions receiving identical probability scores in the supplied file, demonstrating the program’s expected output structure even if specific model runs can produce different values.

This draft manual explains both how a user operates the interface and how the underlying code produces each result.

### Intended audience

This document is written for three overlapping audiences:

- End users who need to run the GUI and export files.
- Technical analysts who need to understand the forecast pipeline and file dependencies.
- Developers or reviewers who need traceability from user-visible controls to code-level behavior.

### What this program is and is not

The program is a research and surveillance support tool. It is not a validated medical device, not a regulatory-grade forecasting engine, and not a replacement for epidemiological judgment. Its main value lies in organizing viral sequence retrieval and generating model-based hypotheses about future strain prominence.

---

## 2. End-to-End Workflow Summary

Before describing each tab separately, it is useful to summarize the entire workflow from input to output.

### Step 1: Bootstrap the environment

When the application starts, it checks whether the NCBI `datasets` and `dataformat` binaries are available either next to the application or on the system path. If they are missing, it downloads the correct binaries for the detected operating system and CPU architecture.

### Step 2: Download viral data

In the Download tab, the user chooses a virus type, an optional geographic filter, an optional release date threshold, and whether only complete sequences should be returned. The program then uses the `datasets` CLI to download an NCBI package archive and extracts either the FASTA file, the metadata CSV, or both.

### Step 3: Collapse duplicate genomes

In the Pattern Machine tab, the user selects a FASTA file. The program parses every sequence entry, cleans the sequence to remove characters outside A, C, G, and T, and groups identical DNA strings together. The first accession in a duplicate group becomes the group’s effective master accession. This produces a JSON file that stores the grouped sequences and identifiers.

### Step 4: Enrich duplicate groups with metadata

If a metadata CSV is supplied in Pattern Machine, the program optionally adds metadata by accession into the JSON structure. For training, the main necessity is not the embedded metadata itself, but the accession mapping that lets later code align the downloaded metadata rows with the deduplicated DNA groups.

### Step 5: Build the supervised learning table

In the forecasting code, the metadata CSV is normalized so columns such as accession, country, date, and title can be recognized even if header names vary. Each row is then mapped from its raw accession to a master accession in the duplicates JSON. Using country and collection date, the program calculates strain frequency by country-month and derives the next-month outbreak label.

### Step 6: Train the CNN

The forecast tab calls `run_strain_growth_forecast`. That function constructs one-hot DNA tensors of fixed length, encodes country identifiers numerically, combines DNA and metadata in a multi-input convolutional neural network, and trains the model using data before the selected target year.

### Step 7: Backtest the held-out year

If rows exist for the target year, the model scores those rows and produces both a standard binary classification backtest against the outbreak label and a dominant-strain backtest that compares the model’s top accession against the actually most frequent accession for each observed country-month.

### Step 8: Generate a global forecast

After training, the model performs a broad scan over a list of countries, years, and months. For each country-month combination, it scores all available master accessions and retains the top N highest-probability strains.

### Step 9: Analyze growth patterns

The Growth Patterns tab reads the forecast CSV and identifies the fastest rising and falling strains as well as the dominant predicted strain by month. This produces a textual analysis view for interpretation.

---

## 3. Platform Bootstrapping and Installation Behavior

One of the most useful implementation details in `ncbi_virus_downloader.py` is that the application hides much of the NCBI command-line setup complexity from the user.

### Startup detection logic

The helper functions `_app_dir`, `_binary_name`, and `_find_binary` determine where the application is running from and whether `datasets` and `dataformat` already exist. The program first checks for local binaries in the application directory. If they are not present, it falls back to `shutil.which` to search the system path.

### Supported operating systems

The `_CLI_URLS` dictionary defines explicit binary download URLs for:

- Windows AMD64 / x86_64
- macOS x86_64
- macOS arm64
- Linux x86_64
- Linux aarch64

This means the desktop application is intended to be cross-platform, even though the observed screenshots and current operating environment are Windows AMD64.

### Automatic binary download

If a binary is missing, `_download_binary` downloads it directly from the NCBI FTP endpoint under `https://ftp.ncbi.nlm.nih.gov/pub/datasets/command-line/v2`. The function reports progress, writes the file into the app directory, and applies execute permissions on non-Windows systems.

### Runtime verification

After bootstrap, the GUI logs:

- platform type
- application directory
- resolved path to `datasets`
- resolved path to `dataformat`
- CLI version from `datasets --version`

This startup phase matters operationally because every later download depends on those binaries. If the bootstrap fails, the buttons in the Download tab remain disabled and the user cannot proceed.

### Practical installation notes for users

From a user perspective, installation is relatively lightweight:

1. Ensure Python and the required Python packages are installed.
2. Launch the Tkinter program.
3. Allow internet access on first run so the NCBI binaries can be fetched if needed.
4. Verify the log says the CLI is ready.

### Practical installation notes for developers

From a developer perspective, there are additional hidden dependencies not handled by the downloader file:

- `pandas`
- `numpy`
- `tensorflow`
- `scikit-learn`
- optionally `tqdm`

These are loaded lazily in `predict_strain_growth.py` by `_load_runtime_deps()`. That design keeps the GUI lightweight at startup, but the forecasting tab still requires a valid machine learning environment.

### Risks during bootstrapping

The main failure modes in this stage are:

- no network connectivity to NCBI
- unsupported platform tuple
- permission issues writing into the application directory
- antivirus or corporate policy blocking executable download
- binary download succeeding but later CLI invocation failing

The log widget is therefore operationally important and should be retained in any future productized version.

**Image placeholder:** Insert `C:\Users\spoca\Desktop\download_1.png` here to illustrate successful startup, tool discovery, and the initial Download tab state.

---

## 4. Download Tab User Guide

The Download tab is the front door into the pipeline. It provides a simplified GUI around the NCBI Datasets virus genome query process.

### Visible controls

Based on the source code and screenshot notes, the tab contains:

- a Virus Type dropdown
- a Location dropdown
- a Released After dropdown
- a Complete sequences only checkbox
- buttons for Download FASTA, Download CSV, and Download Both
- a progress bar
- a status label
- a multi-line log

### Virus type selection

The `VIRUS_TYPES` mapping translates display labels into CLI taxon values. Supported labels include Influenza A, Influenza B, SARS-CoV-2, MERS-CoV, RSV, Ebola, Zika, Dengue, Mpox, HIV-1, hepatitis viruses, and subtype-specific influenza options such as H3N2, H1N1, H5N1, and H7N9.

This mapping is important because the GUI displays user-friendly labels, while the CLI command needs a taxon ID or accepted string token.

### Geographic filter

The Location dropdown corresponds to the `--geo-location` option in the NCBI CLI. It supports country names and broader regions such as Africa, Asia, Europe, North America, South America, and Oceania. If the user leaves it as `Any Location`, the program does not pass a geographic flag.

### Date filter

The Released After control does not ask the user to type a date directly. Instead, `_resolve_date` converts relative choices such as Last 30 days, Last 1 year, or Last 5 years into an absolute ISO-style date string.

### Complete sequences checkbox

If selected, the app appends `--complete-only` to the `datasets` command. This can materially affect dataset size and quality, especially for viruses that have a large mix of partial and complete entries.

### Download actions

The three download buttons have slightly different workflows:

- **Download FASTA** prompts for a single FASTA destination path.
- **Download CSV** prompts for a single CSV destination path.
- **Download Both** prompts for a folder and then creates both output files using a generated base name.

The program names outputs using a timestamped convention such as `ncbi_Influenza_A_virus_YYYYMMDD_HHMMSS`.

### What command is actually run

At the code level, `run_datasets_download()` builds a subprocess command like this in conceptual form:

`datasets download virus genome taxon <taxon> --filename <zip> --no-progressbar [--geo-location <location>] [--released-after <date>] [--complete-only]`

This produces an NCBI package archive, not a direct FASTA or CSV file.

### How outputs are extracted

If a FASTA output is requested, `extract_fasta()` opens the ZIP package and copies `ncbi_dataset/data/genomic.fna` to the requested destination.

If a CSV output is requested, `extract_csv()` uses `dataformat tsv virus-genome --package <zip> --fields ...` and writes the resulting tabular data into a UTF-8 CSV file. The exported fields include:

- accession
- virus-name
- virus-tax-id
- isolate-collection-date
- geo-location
- host-name
- length
- completeness
- release-date
- sourcedb

### Operational behavior

The tab disables the buttons while a job runs, starts an indeterminate progress bar, updates status text, logs the ZIP contents, and shows a message box on success or failure.

### Common user workflow

A typical user sequence is:

1. Choose Influenza A virus or another supported virus.
2. Narrow to a geography such as USA or United Kingdom.
3. Select a release window such as Last 5 years.
4. Decide whether to require complete sequences.
5. Export FASTA and CSV together.
6. Save both files into the same project folder.

### Why this tab matters technically

The Download tab standardizes the file inputs expected downstream. The Pattern Machine expects a FASTA file, and the forecast pipeline expects a metadata CSV with date, location, and accession information. Without this tab, the user would need to manually recreate those outputs using the CLI.

**Image placeholder:** Insert `C:\Users\spoca\Desktop\download_two.png` here to illustrate the Download tab controls and log panel in more detail.

---

## 5. Pattern Machine Tab User Guide and Internal Logic

The Pattern Machine tab is the data deduplication layer of the application.

### Why duplicate detection is necessary

Public viral sequence repositories often contain many accessions whose actual nucleotide strings are identical. If the model naively trained on every duplicate row as if each sequence were biologically distinct, it could waste memory, increase runtime, and overweight common duplicate strings in a way that obscures the core genomic patterns.

The Pattern Machine addresses this by grouping identical cleaned DNA sequences into sequence groups.

### Inputs and controls

The tab contains:

- FASTA Input File
- Metadata CSV (optional)
- Embed metadata in duplicates JSON checkbox
- Max Sequences input
- Find Duplicates button
- progress bar
- status line
- results summary panel
- log panel

### FASTA parsing

The function `parse_fasta_entries()` reads the FASTA file incrementally. For each header line beginning with `>`, it extracts the identifier using `_extract_fasta_id()`, which keeps the first token of the header as the accession-like identifier. Sequence lines are concatenated until the next header.

### DNA cleaning

The helper `_clean_dna()` applies the regex `[^ACGT]` and removes any character not in A, C, G, or T after converting the sequence to uppercase. This is a strong normalization step. Ambiguous bases such as N are effectively dropped here in the duplicate-building phase.

That choice has two implications:

- It makes duplicate grouping stricter around canonical DNA letters.
- It may collapse sequences that differ only in ambiguous characters into the same cleaned sequence.

### Duplicate grouping

`build_duplicate_dict()` uses a `defaultdict(list)` keyed by cleaned sequence string. Each key maps to a list of identifiers sharing that exact sequence. It records:

- the sequence string
- duplicate count
- all identifiers in the group

It then sorts the groups by count and sequence length, producing a JSON structure with:

- `source_file`
- `max_seqs`
- `total_sequences`
- `unique_sequences`
- `duplicate_sequences`
- `sequences`

Each sequence entry contains the sequence, count, and identifiers.

### Metadata enrichment

If the user also supplies a metadata CSV, the program reads it into a dictionary keyed by `Accession`. For each duplicate group, it builds `metadata_by_accession`, allowing the JSON to preserve metadata per original accession.

This enrichment is optional because the forecast training code does not strictly require the metadata to be embedded in the JSON file itself; it mainly needs identifier mapping from accessions to the master accession. Still, the enriched JSON is helpful for manual review and auditing.

### Result metrics shown to the user

The tab reports at least:

- total sequences processed
- unique sequences remaining after collapse
- duplicate rows detected

In the supplied screenshot analysis, an example run reported over 1.58 million total sequences, approximately 895 thousand unique sequences, and roughly 686 thousand duplicate rows. That demonstrates the scale reduction this step can provide.

### Output significance

This JSON file is a required or near-required bridge into the forecasting stage because `predict_strain_growth.py` uses it to:

- map raw accessions onto a master accession
- construct one-hot DNA tensors by master accession
- avoid re-encoding duplicate DNA unnecessarily

### User instructions

Recommended user procedure:

1. Select the FASTA generated from the Download tab.
2. Optionally select the corresponding metadata CSV.
3. Leave Max Sequences blank unless performing a test run.
4. If training efficiency is the goal, leave metadata embedding off.
5. Click Find Duplicates.
6. Save the generated JSON and note the reported totals.

### Technical caveat

The metadata loader here requires an `Accession` column exactly. By contrast, the forecast script has more flexible header normalization. This means a CSV exported directly from the downloader may need either consistent naming or later normalization. In practice, the forecast script is more tolerant than the Pattern Machine metadata enricher.

**Image placeholder:** Insert `C:\Users\spoca\Desktop\pattern_machine.png` here to show the duplicate-detection interface and example summary metrics.

---

## 6. Strain Forecast Tab User Guide

The Strain Forecast tab connects the prepared inputs to the machine learning engine in `predict_strain_growth.py`.

### User-facing inputs

The tab exposes the following fields:

- Metadata CSV
- Duplicates JSON
- Output Forecast CSV
- Countries (comma-separated)
- Forecast Years
- Training Years
- Backtest Year
- Dominant Year
- Epochs
- Batch Size
- Top N
- Run Strain Forecast button

This is the highest-leverage tab in the application because it controls training, backtesting, and inference behavior.

### Expected files

The Metadata CSV should contain accession, collection date, and country information. The Duplicates JSON should contain grouped sequences with identifiers. The output file path is where the forecast CSV will be written.

### Country list

The countries field supports a comma-separated set such as:

`USA, Canada, France, Mexico, United Kingdom`

This list constrains the global forecast scan and the dominant-strain backtest. It is not just a display filter; it materially affects runtime and output size.

### Forecast years versus training years

These two fields should not be conflated.

- `Forecast Years` define the years for which the model will generate scored outputs in the final report.
- `Training Years` define which historical years are eligible for model fitting, subject to exclusion of the target year.

The helper `parse_years()` supports both comma-separated values and ranges such as `2020-2025`.

### Backtest year and dominant year

The code sets `target_year = dominant_year or backtest_year`. That means the dominant-year setting, if specified, overrides the backtest target. The model then trains on data before that target year and tests on the rows from that year.

### Epochs and batch size

Epochs control how many passes the model makes over the training data. Batch size is used both in model training and in chunked prediction when scanning accessions.

### Top N

For each country-month in the global forecast, the model keeps only the top N highest-probability accessions. A higher Top N yields a larger forecast CSV and more alternatives for analysis in the Growth Patterns tab.

### Runtime feedback

The log and status output are important because the forecast may run for a long time. The source code reports:

- dataset build completion
- training-year filtering
- per-epoch loss and accuracy
- R2 backtest summary
- classification report text
- dominant backtest summary
- per-country-month scan progress
- final output path

### Recommended usage pattern

For a first serious run:

1. Use a cleaned metadata CSV from the Download tab.
2. Use the duplicate JSON from Pattern Machine.
3. Keep countries limited to a few regions initially.
4. Use a small epoch count for smoke testing, then increase for fuller runs.
5. Select a clear held-out backtest year such as 2025.
6. Set Top N to 3 or 5 for manageable output size.

### Output products

This stage can create two distinct CSV files:

- the main forecast output CSV
- a dominant backtest CSV named like `<stem>_dominant_backtest_<year>.csv`

The dominant backtest file appears only if comparable rows exist.

**Image placeholder:** Insert `C:\Users\spoca\Desktop\strain_forecast.png` here to show the forecast configuration panel and in-progress scanning log.

---

## 7. Growth Patterns Tab User Guide

The Growth Patterns tab is the post-forecast interpretation layer. It does not train the model; it reads the forecast output and summarizes directional changes.

### Purpose

This tab helps answer questions like:

- Which predicted strains are rising fastest?
- Which are falling?
- Which accession is predicted to dominate in a country for a given month?
- How many countries and strains are represented in the loaded file?

### Inputs and controls

From the screenshot analysis, the tab includes:

- Forecast Output CSV selector
- Country Filter
- Top Trends numeric input
- Analyze Growth Patterns button
- status line
- report panel for rising/falling strains
- report panel for dominant monthly predictions

### Analytical outputs

The first report ranks strains using metrics such as:

- Start probability
- End probability
- Delta
- Peak probability and timing

The second report shows dominant predictions by month with columns such as:

- Country
- Date
- Accession
- Probability
- Title

### Relationship to the forecast file

The supplied CSV `global_outbreak_forecast_2020_2025new.csv` follows the expected structure, containing rows like:

- `2020,1,France,JF701833.1,0.480342..., Influenza A virus ...`

This is exactly the kind of data the Growth Patterns tab is designed to summarize.

### Practical interpretation

The tab does not make a new prediction; it helps a user compare the predicted trajectory of strains over time. A strain with a large positive delta is interpreted as rising risk or rising expected prominence. A dominant monthly prediction can serve as a shortlist candidate for closer laboratory, surveillance, or literature review.

### User procedure

1. Load the forecast CSV generated in the Strain Forecast tab.
2. Leave Country Filter as All Countries for a global overview, or restrict it to one country.
3. Choose how many top trends to display.
4. Click Analyze Growth Patterns.
5. Review the textual reports.

### Current limitations of the tab

The growth analysis is textual rather than graphical. It surfaces trends but does not explain why the model prefers one accession over another.

**Image placeholder:** Insert `C:\Users\spoca\Desktop\analyze_growth_patterns.png` here to show the post-forecast trend and dominant-prediction reports.

---

## 8. Metadata Normalization and Master Dataset Construction

The technical heart of the forecasting pipeline begins in `build_master_dataset()` inside `predict_strain_growth.py`.

### Flexible metadata column recognition

The helper `_first_existing_column()` and `normalize_metadata_columns()` make the pipeline more robust to differing CSV schemas. The code searches for equivalent names for the key fields:

- Accession / accession
- Collection_Date / isolate-collection-date / collection-date variants
- Country / geo-location / Geographic Location variants
- GenBank title variants

This is essential because the downloader exports NCBI-style headers such as `isolate-collection-date` and `geo-location`, while other files might use normalized internal names.

### Country cleanup

Country is normalized by splitting on the first colon and taking the left side. That means a metadata value like `USA: California` becomes `USA`. This is a meaningful preprocessing decision because it intentionally collapses subnational granularity into country-level modeling.

### DNA tensor creation

The duplicates JSON contains sequence groups. For each group:

- the first identifier becomes the master accession
- all identifiers in the group are mapped to that master accession
- the sequence is one-hot encoded into a fixed `(2500, 4)` tensor

This creates two core lookup objects:

- `accession_to_master`
- `dna_tensors`

### Temporal feature creation

Once metadata rows are loaded and mapped to a master accession, the program converts collection dates to datetime and derives:

- `YearMonth`
- `Year`
- `Month`

These are the temporal anchors for both label creation and downstream forecast conditioning.

### Frequency calculation

The code computes total observations per country-month, then count per country-month-master accession, and divides them to get `Freq`.

Mathematically:

`Freq = Count / Total`

This is the current share of that strain within the observed country-month sample.

### Lookahead construction

The dataset is sorted by country, master accession, and year-month. Then the next month’s frequency for each country-strain series is created using a grouped shift:

`Next_Month_Freq = groupby([Country, Master_Accession])[Freq].shift(-1)`

This is the most important supervisory signal engineering step. The model is not asked merely to reproduce current frequency. It is asked to learn whether present DNA and metadata conditions foreshadow growth in the next month.

### Final training table

The outbreak label is merged back onto the original per-accession metadata rows. Any rows lacking a valid outbreak label are dropped. Finally, country names are label-encoded to integer `Country_ID` values.

The result is a trainable table in which each row contains:

- accession-derived master DNA tensor
- country ID
- month number
- outbreak label

---

## 9. Outbreak Label Logic

The outbreak label is the main ground truth used by the binary classifier. It deserves explicit documentation because it determines what the model is actually learning.

### Definition

A row receives:

- `Outbreak = 1` if the same strain’s frequency in the next month is at least 5%
- `Outbreak = 0` otherwise

In code:

`growth_df["Outbreak"] = (growth_df["Next_Month_Freq"] >= 0.05).astype(int)`

### Interpretation

This label does not mean that the strain is currently dominant. It means the strain is expected, based on future observed data, to reach at least a modest prevalence threshold in the next monthly time step.

So the model is learning an early warning signal rather than a same-period prevalence estimate.

### Why 5%?

The threshold of 5% is a heuristic cut point encoded directly in the script. It converts a continuous future frequency target into a binary event definition. That has both advantages and tradeoffs:

Advantages:

- easier binary classification setup
- simpler user interpretation
- direct linkage to “risk of meaningful appearance next month”

Tradeoffs:

- threshold is arbitrary and may not match public health significance equally across viruses
- small sampling fluctuations around 5% can flip labels
- dominant strains far above 5% are treated the same as marginal strains just above 5%

### Granularity of the label

The label is computed at the master-accession, country, year-month level, then merged onto all metadata rows matching that group. This can replicate the same outbreak label across multiple rows tied to the same master accession and month.

### Relationship to dominant-strain evaluation

The outbreak label supports the binary classifier evaluation. The dominant-strain backtest is related but separate. One asks, “Will this strain clear the 5% next-month threshold?” The other asks, “Which strain will be the top observed one in this country-month?”

That distinction should be emphasized to users, because the main forecast CSV lists top probabilities, but those probabilities originate from the outbreak classifier rather than from a direct multiclass dominant-strain model.

---

## 10. Model Training Pipeline

The forecasting model is a late-fusion, multi-input convolutional neural network.

### Input stream 1: DNA sequence

The DNA stream begins with one-hot encoding. Each position up to length 2500 is encoded as four channels corresponding to A, C, G, and T. Unknown base `N` is represented as zeros in `one_hot_encode_dna`, while longer sequences are truncated and shorter ones are zero-padded.

This gives a tensor shape of `(2500, 4)` per sequence.

### Input stream 2: metadata

The metadata stream consists of two numeric values:

- `Country_ID`
- `Month`

This is a compact metadata representation. It does not currently include host, sequence length, completeness, release date, or database source, even though some of those can be carried through metadata normalization.

### DNA branch architecture

The DNA branch is:

- `Conv1D(32, kernel_size=4, activation='relu')`
- `MaxPooling1D(2)`
- `Conv1D(64, kernel_size=8, activation='relu')`
- `MaxPooling1D(2)`
- `Flatten()`

This branch is designed to act as a motif detector, scanning along the nucleotide sequence to identify short local patterns associated with later growth.

### Metadata branch architecture

The metadata branch is simple:

- `Input(shape=(2,))`
- `Dense(16, activation='relu')`

This allows the network to learn a nonlinear transformation of country and month context before fusion.

### Fusion and classifier head

The outputs of the DNA and metadata branches are concatenated, then passed through:

- `Dense(64, activation='relu')`
- `Dropout(0.5)`
- `Dense(1, activation='sigmoid')`

The final sigmoid output is interpreted as an outbreak probability.

### Optimization

The model is compiled with:

- optimizer: Adam
- loss: binary cross-entropy
- metric: accuracy

### Train-validation handling

If the training set has at least 10 rows, the code uses a `validation_split` of 0.1 during fitting. Otherwise, validation is skipped. A callback reports loss and accuracy after every epoch.

### Training data segregation

The main split is year-based, not random. If the target year is 2025, all rows before 2025 are used for training, and rows from 2025 form the held-out evaluation set.

This is appropriate for temporal forecasting because it avoids leakage from future observations into training.

### Important implication

Although the model is described as predicting dominant strains, its native learning objective is still the binary next-month outbreak label. Dominance is approximated later by selecting the accession with the maximum predicted outbreak probability within a country-month scoring pass.

---

## 11. Backtesting Methodology

The code performs two different forms of backtesting.

### 11.1 Binary outbreak backtest

If held-out rows exist for the target year, the model predicts probabilities for those rows and thresholds them at 0.5. It then produces:

- a `classification_report` for Normal versus Outbreak
- an R2 score comparing predicted probabilities against binary outcomes

The inclusion of R2 is unconventional for binary classification but still provides a rough sense of calibration or continuous alignment.

### 11.2 Dominant strain backtest

The function `run_dominant_strain_backtest()` provides a more operational evaluation.

#### Ground truth generation

For the target year, it groups observed metadata by:

- Country
- Month
- Master_Accession

It counts rows and identifies the highest-count accession for each country-month as the actual dominant strain.

#### Prediction generation

For each observed country-month in the target year, the code:

- transforms the country into `Country_ID`
- scores every available master accession at that month
- selects the accession with the maximum predicted probability

That accession becomes the predicted dominant strain.

#### Recorded fields

The dominant backtest output includes:

- Year
- Month
- Country
- Predicted_Accession
- Predicted_Probability
- Actual_Accession
- Previous_Actual_Dominant_Accession
- Actual_Count
- Actual_New_DNA_From_Previous_Dominant
- Match
- Previous_Predicted_Dominant_Accession
- Predicted_New_DNA_From_Previous_Dominant
- Predicted_Title
- Actual_Title

#### Summary metric

The printed summary is exact-match accuracy across country-months.

This is a stricter and more intuitive metric for end users than binary outbreak accuracy because it directly asks whether the top forecasted accession matched the actually most common accession.

### New DNA shift detection

Both the forecast annotations and the dominant backtest track whether the dominant predicted accession changes relative to the prior month. This is a valuable signal because it highlights transitions in likely circulating DNA, not just persistence of the same strain.

### Methodological strength

The backtest is time-aware and country-aware, which is appropriate.

### Methodological limitation

The dominant-strain backtest still relies on a model trained for outbreak classification, not for direct dominance ranking. So exact-match accuracy should be interpreted as a secondary derived evaluation, not the objective the network was optimized to maximize.

---

## 12. Forecast Output Structure and How to Read It

The main forecast output is a CSV produced by `run_global_forecast()`.

### Core fields

The source code and supplied file show these core columns:

- `Year`
- `Month`
- `Country`
- `Accession`
- `Probability`
- `GenBank_Title`

The source code can also append annotation columns related to dominant prediction and DNA shifts, including:

- `Dominant_Accession`
- `Previous_Dominant_Accession`
- `Previous_Dominant_Title`
- `New_DNA_From_Previous_Dominant`
- `Is_Dominant_Prediction`

### Meaning of probability

The probability is the model’s sigmoid output for the outbreak label, conditioned on a specific country and month. It should be interpreted as the model’s confidence that the accession is associated with next-month growth above the defined threshold, not as a literal incidence rate.

### Top N behavior

For every country-month, only the top N highest probabilities are retained. Therefore, the file is not a full exhaustive matrix of all accessions unless Top N was set very high.

### Reading the supplied sample file

The supplied `global_outbreak_forecast_2020_2025new.csv` contains early rows such as France in January through April 2020, each listing several H3N2-related accessions with the same probability value. This suggests either a tied model output in that particular run or a forecasting scenario where many accessions received nearly indistinguishable scores.

### How analysts should use the CSV

The main use cases are:

- identify candidate dominant strains by month
- compare countries for likely accession shifts
- feed the Growth Patterns tab
- manually inspect high-risk accessions and titles
- track when new dominant DNA is predicted to emerge

### Secondary output: dominant backtest CSV

If produced, the dominant backtest CSV should be interpreted as an evaluation artifact rather than a forecast artifact. It is valuable for model validation and reporting.

---

## 13. Mapping the GUI Tabs to the Codebase

One useful way to understand the full program is to align each GUI tab with the exact code responsibilities.

### Download tab to code

Main code regions in `ncbi_virus_downloader.py`:

- CLI URL table and platform detection
- `ensure_cli()`
- `run_datasets_download()`
- `extract_fasta()`
- `extract_csv()`
- `_build_download_tab()`
- `_on_dl()` and `_dl_thread()`

### Pattern Machine tab to code

Main code regions:

- `parse_fasta_entries()`
- `_clean_dna()`
- `load_metadata_by_accession()`
- `build_duplicate_dict()`
- `_build_pattern_tab()` and its callbacks

### Strain Forecast tab to code

Forecasting engine in `predict_strain_growth.py`:

- `normalize_metadata_columns()`
- `build_master_dataset()`
- `prepare_gpu_inputs()`
- `build_multi_input_cnn()`
- `run_dominant_strain_backtest()`
- `run_global_forecast()`
- `run_strain_growth_forecast()`

The GUI tab itself appears to be built in the latter portion of `ncbi_virus_downloader.py`, where the application creates all four tabs, though the main forecasting logic lives in the second script.

### Growth Patterns tab to code

The screenshot and tab presence confirm a Growth Patterns analysis view exists in the GUI. The supplied source excerpts show the tab creation hook in `App.__init__`, but the specific implementation body is not fully visible in the excerpt reviewed here. Even so, the screenshot analysis clearly shows the feature set and the forecast CSV semantics confirm how it is expected to operate.

### Architectural takeaway

Although split across two files, the application behaves as one integrated workbench. The downloader file is not just a downloader; it is the main GUI shell containing all tabs. The forecasting script is the ML engine invoked from that shell.

---

## 14. Limitations and Known Technical Risks

This system is useful, but several important limitations should be documented explicitly.

### 14.1 DNA cleaning can discard ambiguity information

The duplicate builder removes any character outside A, C, G, and T. Ambiguous symbols can carry sequencing uncertainty or biologically relevant ambiguity, but they are erased before deduplication.

### 14.2 Fixed sequence length may truncate informative regions

The model only uses the first 2500 bases. Longer genomes are truncated, which can discard distal motifs. Shorter sequences are padded, which may also affect learned patterns.

### 14.3 Metadata input is minimal

The model metadata branch uses only country ID and month. It ignores host, sequence completeness, release timing, sequence length, lineage annotations, and many other potentially predictive features already available or derivable.

### 14.4 Country-level aggregation is coarse

Subnational resolution is collapsed by splitting `geo-location` at the first colon. That improves consistency but sacrifices regional detail.

### 14.5 Outbreak threshold is heuristic

The 5% next-month threshold is hard-coded. It may be too low for some settings and too high for others.

### 14.6 Dominant prediction is indirect

The system chooses the highest outbreak probability as the dominant strain forecast. This is sensible as a heuristic, but it is not the same as training a model directly to predict dominance.

### 14.7 Backtest depends on metadata representativeness

Observed dominant strain counts are based on sampled metadata rows, not necessarily true population prevalence. Biases in sequence submission can distort both training and evaluation.

### 14.8 No explicit phylogenetic modeling

The model processes raw sequence motifs but does not explicitly encode evolutionary relationships, clades, or substitution history.

### 14.9 Potential computational load

Scoring all accessions for every country-month can be expensive when the duplicate JSON contains many unique sequences.

### 14.10 Limited interpretability

The current output does not reveal which motifs drove a high-risk score. Users get probabilities and accession names, but not explanatory sequence features.

---

## 15. Recommended Future Interpretability and Product Improvements

The parent task specifically requests recommended future interpretability improvements. This is a high-priority enhancement area for the current architecture.

### 15.1 Integrated gradients or saliency maps

The most direct improvement would be to add attribution methods that highlight nucleotide positions contributing most to a high outbreak score. With this, the model could output not just a probability but also a ranked set of influential positions or short motifs.

### 15.2 Conv1D filter visualization

Because the first convolutional layers use kernel widths 4 and 8, their learned weights can be inspected as candidate motif detectors. Exporting and summarizing the strongest filters could provide a first-pass “motif lexicon” of what the model considers risky.

### 15.3 Attention or explainable fusion

A future model could replace or augment the current fusion mechanism with attention layers that make it clearer when geography versus sequence content is driving a prediction.

### 15.4 Calibration analysis

Probability calibration plots would help users understand whether a score like 0.48 is truly meaningful across countries and years.

### 15.5 Direct dominant-strain objective

A second model head could be trained directly for dominant accession ranking while the existing head predicts outbreak threshold crossing. That would better align the training objective with the dominant-strain backtest.

### 15.6 Additional metadata channels

Features worth adding include:

- host species
n- completeness
- segment/gene information
- lineage or subtype labels
- sequence length
- release lag
- region beyond country

These could improve both predictive power and interpretability.

### 15.7 Better uncertainty reporting

Instead of a single point probability, the software could provide uncertainty intervals via ensembling, Monte Carlo dropout, or repeated training seeds.

### 15.8 Visual analytics in Growth Patterns

The current Growth Patterns tab is text-centric. Adding time-series plots, country comparison charts, and accession transition diagrams would make results more interpretable.

### 15.9 Data lineage traceability

Every forecast row should ideally be traceable back to:

- source FASTA file
- source metadata CSV
- duplicate JSON version
- model hyperparameters
- training years and target year
- timestamp and software version

### 15.10 Reproducibility and model packaging

Future versions should save the fitted model, encoders, and preprocessing schema so that a forecast can be reproduced exactly without retraining.

---

## 16. Practical User Procedure: Recommended Standard Operating Flow

This section condenses the full manual into a practical runbook.

### Phase A: Acquire data

1. Open the application.
2. Wait for CLI setup to complete.
3. In Download, choose the virus, location, date filter, and completeness option.
4. Export both FASTA and CSV into the same project folder.

### Phase B: Build the duplicates JSON

1. Open Pattern Machine.
2. Select the FASTA file from Phase A.
3. Optionally select the matching metadata CSV.
4. Leave Max Sequences blank for full production runs.
5. Click Find Duplicates and keep the JSON output.

### Phase C: Run the forecast

1. Open Strain Forecast.
2. Select the metadata CSV.
3. Select the duplicates JSON.
4. Choose the output forecast CSV destination.
5. Enter the country list.
6. Set forecast years, training years, backtest year, epoch count, batch size, and Top N.
7. Run the forecast and review the log.
8. Save both the main forecast CSV and, if present, the dominant backtest CSV.

### Phase D: Interpret outputs

1. Open Growth Patterns.
2. Load the forecast CSV.
3. Apply country filter if needed.
4. Review the rising/falling and dominant monthly reports.
5. Cross-check high-probability accessions manually where needed.

---

## 17. Developer Notes and Suggested Documentation Gaps to Fill Later

Because this deliverable is a prose draft based on source review rather than a full instrumented execution of the entire GUI, several follow-up documentation improvements are recommended.

### Areas that should be validated in a future revision

- confirm the exact Growth Patterns implementation logic from the remaining GUI source
- document the Strain Forecast tab’s specific callback code and GUI validation behavior
- add a dependency installation section with exact package versions
- include a troubleshooting appendix for common TensorFlow and GPU issues
- capture one complete real run from download through forecast and record all intermediate file names

### Suggested appendices for a final manual

- sample CLI command lines emitted by the Download tab
- sample duplicate JSON schema excerpt
- sample dominant backtest CSV excerpt
- package installation instructions for Windows, macOS, and Linux
- glossary of accession, master accession, outbreak, dominant strain, and delta

---

## 18. Conclusion

The viral forecasting program is best understood as an integrated research pipeline rather than a single prediction function. Its strength is that it turns a complicated workflow into a sequence of manageable, tab-driven steps. The Download tab acquires data from NCBI. Pattern Machine reduces redundancy and creates a sequence-group representation. The Strain Forecast tab converts grouped DNA plus metadata into a trainable temporal forecasting problem and then generates a country-month accession ranking. The Growth Patterns tab turns those forecasts into analyst-readable trend summaries.

From a technical perspective, the key innovation is the combination of genomic sequence pattern extraction with simple temporal and geographic context. From a practical perspective, the most important caution is that the model predicts a next-month outbreak proxy based on a 5% threshold, and only indirectly infers dominant strains by ranking probabilities.

Used carefully, the system can support hypothesis generation, surveillance prioritization, and retrospective performance analysis. To become more scientifically persuasive and operationally transparent, its next major improvements should focus on interpretability, calibration, richer metadata usage, and clearer reproducibility.

---

## Image Placement Summary

For the final formatted manual, insert the supplied screenshots at the following points:

1. `C:\Users\spoca\Desktop\download_1.png`
   - Place in Section 3 or early Section 4 to show startup readiness and Download tab overview.

2. `C:\Users\spoca\Desktop\download_two.png`
   - Place in Section 4 to show Download tab controls and download workflow.

3. `C:\Users\spoca\Desktop\pattern_machine.png`
   - Place in Section 5 to show duplicate analysis inputs and output summary.

4. `C:\Users\spoca\Desktop\strain_forecast.png`
   - Place in Section 6 to show forecast configuration and scan progress.

5. `C:\Users\spoca\Desktop\analyze_growth_patterns.png`
   - Place in Section 7 to show growth analysis reports and dominant monthly predictions.
