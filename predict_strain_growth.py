import json
import os
from pathlib import Path

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

MAX_SEQ_LENGTH = 2500


def _progress(iterable, progress_callback=None, **kwargs):
    if progress_callback or tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _log(progress_callback, message, percent=None):
    if progress_callback:
        try:
            progress_callback(message, percent)
        except TypeError:
            progress_callback(message)
    else:
        print(message)


def _load_runtime_deps():
    """Import heavy ML dependencies only when a forecast is actually run."""
    os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

    import numpy as np
    import pandas as pd
    try:
        import tensorflow as tf
    except Exception as exc:
        raise RuntimeError(
            "TensorFlow is not installed correctly in this Python environment. "
            "Please reinstall it, then rerun forecast.\n"
            "Suggested fix:\n"
            "  python -m pip uninstall -y tensorflow tensorflow-intel keras\n"
            "  python -m pip install --upgrade pip\n"
            "  python -m pip install tensorflow"
        ) from exc
    from sklearn.metrics import classification_report
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import LabelEncoder

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            print(f"Memory growth error: {exc}")

    from tensorflow.keras.callbacks import Callback
    from tensorflow.keras.layers import (
        Concatenate,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        Input,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model

    return {
        "np": np,
        "pd": pd,
        "classification_report": classification_report,
        "r2_score": r2_score,
        "LabelEncoder": LabelEncoder,
        "Callback": Callback,
        "Concatenate": Concatenate,
        "Conv1D": Conv1D,
        "Dense": Dense,
        "Dropout": Dropout,
        "Flatten": Flatten,
        "Input": Input,
        "MaxPooling1D": MaxPooling1D,
        "Model": Model,
    }


def _first_existing_column(df, candidates):
    def normalize_header(value):
        return str(value).strip().lstrip("\ufeff").casefold()

    by_lower = {normalize_header(col): col for col in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        found = by_lower.get(normalize_header(candidate))
        if found:
            return found
    return None


def normalize_metadata_columns(df):
    """Normalize downloaded NCBI CSV columns for the forecast pipeline."""
    accession_col = _first_existing_column(df, ["Accession", "accession"])
    date_col = _first_existing_column(
        df,
        [
            "Collection_Date",
            "isolate-collection-date",
            "Isolate Collection date",
            "collection-date",
            "Isolate Collection Date",
        ],
    )
    country_col = _first_existing_column(
        df,
        [
            "Country",
            "geo-location",
            "Geo_Location",
            "Geo Location",
            "Geographic Location",
        ],
    )
    title_col = _first_existing_column(
        df, ["GenBank_Title", "virus-name", "Virus Name", "title"]
    )
    optional_columns = {
        "Host_Name": ["host-name", "Host Name", "host"],
        "Length": ["length", "Length"],
        "Completeness": ["completeness", "Completeness"],
        "Release_Date": ["release-date", "Release date", "Release Date"],
        "Source_Database": ["sourcedb", "Source database", "Source Database"],
    }

    missing = []
    if accession_col is None:
        missing.append("Accession/accession")
    if date_col is None:
        missing.append("Collection_Date/isolate-collection-date/Isolate Collection date")
    if country_col is None:
        missing.append("Country/geo-location/Geographic Location")
    if missing:
        available = ", ".join(str(col).strip().lstrip("\ufeff") for col in df.columns)
        raise ValueError(
            "Metadata CSV is missing required columns: "
            + ", ".join(missing)
            + f". Available columns: {available}"
        )

    normalized = df.copy()
    normalized["Accession"] = normalized[accession_col].astype(str)
    normalized["Collection_Date"] = normalized[date_col]
    normalized["Country"] = (
        normalized[country_col]
        .astype(str)
        .str.split(":", n=1)
        .str[0]
        .str.strip()
    )
    if title_col:
        normalized["GenBank_Title"] = normalized[title_col]
    elif "GenBank_Title" not in normalized.columns:
        normalized["GenBank_Title"] = normalized["Accession"]
    for output_col, candidates in optional_columns.items():
        source_col = _first_existing_column(df, candidates)
        if source_col:
            normalized[output_col] = normalized[source_col]
    return normalized


def one_hot_encode_dna(sequence, max_len=MAX_SEQ_LENGTH, np_module=None):
    np = np_module or _load_runtime_deps()["np"]
    mapping = {
        "A": [1, 0, 0, 0],
        "C": [0, 1, 0, 0],
        "G": [0, 0, 1, 0],
        "T": [0, 0, 0, 1],
        "N": [0, 0, 0, 0],
    }
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    for i, char in enumerate(sequence[:max_len]):
        if char in mapping:
            encoded[i] = mapping[char]
    return encoded


def build_master_dataset(csv_path, json_path, progress_callback=None, deps=None):
    deps = deps or _load_runtime_deps()
    pd = deps["pd"]
    LabelEncoder = deps["LabelEncoder"]

    _log(progress_callback, f"Loading DNA sequences from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    accession_to_master, dna_tensors = {}, {}
    sequence_groups = data.get("sequences", [])
    for group in _progress(
        sequence_groups,
        progress_callback,
        desc="Encoding DNA sequences",
        unit="seq",
    ):
        idents = group.get("identifiers", [])
        if not idents:
            continue
        master = str(idents[0])
        for ident in idents:
            accession_to_master[str(ident)] = master
        dna_tensors[master] = one_hot_encode_dna(
            group.get("sequence", "").upper(), np_module=deps["np"]
        )

    _log(progress_callback, f"Processing metadata from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = normalize_metadata_columns(df)
    df["Master_Accession"] = df["Accession"].map(accession_to_master)
    df = df.dropna(subset=["Master_Accession", "Collection_Date", "Country"])
    df["Collection_Date"] = pd.to_datetime(df["Collection_Date"], errors="coerce")
    df["YearMonth"] = df["Collection_Date"].dt.to_period("M")
    df["Year"] = df["Collection_Date"].dt.year
    df["Month"] = df["Collection_Date"].dt.month

    math_df = df.dropna(subset=["YearMonth"]).copy()
    monthly_totals = (
        math_df.groupby(["Country", "YearMonth"]).size().reset_index(name="Total")
    )
    strain_monthly = (
        math_df.groupby(["Country", "YearMonth", "Master_Accession"])
        .size()
        .reset_index(name="Count")
    )
    growth_df = strain_monthly.merge(monthly_totals, on=["Country", "YearMonth"])
    growth_df["Freq"] = growth_df["Count"] / growth_df["Total"]

    growth_df = growth_df.sort_values(by=["Country", "Master_Accession", "YearMonth"])
    growth_df["Next_Month_Freq"] = growth_df.groupby(
        ["Country", "Master_Accession"]
    )["Freq"].shift(-1)
    growth_df["Outbreak"] = (growth_df["Next_Month_Freq"] >= 0.05).astype(int)

    final_df = df.merge(
        growth_df[["Country", "YearMonth", "Master_Accession", "Outbreak"]],
        on=["Country", "YearMonth", "Master_Accession"],
        how="left",
    ).dropna(subset=["Outbreak"])

    if final_df.empty:
        raise ValueError(
            "No trainable rows were produced. Check that CSV accessions match "
            "the duplicate JSON identifiers and include dated country metadata."
        )

    country_le = LabelEncoder()
    final_df["Country_ID"] = country_le.fit_transform(final_df["Country"])
    return final_df, dna_tensors, country_le


def prepare_gpu_inputs(df, dna_tensors, np_module=None, progress_callback=None):
    np = np_module or _load_runtime_deps()["np"]
    X_dna, X_meta, y = [], [], []
    for _, row in _progress(
        df.iterrows(),
        progress_callback,
        total=len(df),
        desc="Preparing model inputs",
        unit="row",
    ):
        master = row["Master_Accession"]
        if master in dna_tensors:
            X_dna.append(dna_tensors[master])
            X_meta.append([row["Country_ID"], row["Month"]])
            y.append(row["Outbreak"])
    return np.array(X_dna), np.array(X_meta), np.array(y)


def build_multi_input_cnn(deps=None):
    deps = deps or _load_runtime_deps()
    Concatenate = deps["Concatenate"]
    Conv1D = deps["Conv1D"]
    Dense = deps["Dense"]
    Dropout = deps["Dropout"]
    Flatten = deps["Flatten"]
    Input = deps["Input"]
    MaxPooling1D = deps["MaxPooling1D"]
    Model = deps["Model"]

    dna_in = Input(shape=(MAX_SEQ_LENGTH, 4), name="DNA_Input")
    x = Conv1D(32, 4, activation="relu")(dna_in)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 8, activation="relu")(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    meta_in = Input(shape=(2,), name="Meta_Input")
    m = Dense(16, activation="relu")(meta_in)
    combined = Concatenate()([x, m])
    z = Dense(64, activation="relu")(combined)
    z = Dropout(0.5)(z)
    out = Dense(1, activation="sigmoid")(z)
    model = Model(inputs=[dna_in, meta_in], outputs=out)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def iter_dna_chunks(accessions, dna_tensors, np_module, chunk_size):
    chunk_size = max(1, int(chunk_size or 1))
    for start in range(0, len(accessions), chunk_size):
        chunk_accessions = accessions[start:start + chunk_size]
        dna_batch = np_module.array([dna_tensors[acc] for acc in chunk_accessions])
        yield chunk_accessions, dna_batch


def predict_for_all_accessions(
    model,
    accessions,
    dna_tensors,
    meta_values,
    np_module,
    prediction_batch_size,
):
    predictions_by_accession = {}
    chunk_size = max(1, int(prediction_batch_size or 1))
    for chunk_accessions, dna_batch in iter_dna_chunks(
        accessions, dna_tensors, np_module, chunk_size
    ):
        meta_batch = np_module.array([meta_values] * len(chunk_accessions))
        probabilities = model.predict(
            [dna_batch, meta_batch],
            batch_size=chunk_size,
            verbose=0,
        ).flatten()
        predictions_by_accession.update(zip(chunk_accessions, probabilities))
    return predictions_by_accession


def run_global_forecast(
    model,
    dna_tensors,
    country_le,
    metadata_df,
    target_countries=None,
    years=None,
    top_n=3,
    prediction_batch_size=256,
    progress_callback=None,
    deps=None,
    progress_start=75,
    progress_end=95,
):
    deps = deps or _load_runtime_deps()
    np = deps["np"]
    pd = deps["pd"]
    target_countries = target_countries or [
        "USA",
        "Canada",
        "France",
        "Mexico",
        "United Kingdom",
    ]
    years = years or [2020, 2021, 2022, 2023, 2024, 2025]
    accessions = list(dna_tensors.keys())
    if not accessions:
        raise ValueError("No DNA tensors were available for forecasting.")
    forecast_results = []

    total_steps = max(1, len(target_countries) * len(years) * 12)
    completed_steps = 0
    scan_bar = None
    if progress_callback is None and tqdm is not None:
        scan_bar = tqdm(total=total_steps, desc="Forecast scan", unit="month")

    try:
        _log(progress_callback, "Initiating global threat scan...", progress_start)
        for country in target_countries:
            try:
                c_id = country_le.transform([country])[0]
            except ValueError:
                _log(progress_callback, f"Skipping {country}: no historical training data.")
                if scan_bar is not None:
                    scan_bar.update(len(years) * 12)
                continue
            for year in years:
                for month in range(1, 13):
                    predictions_by_accession = predict_for_all_accessions(
                        model,
                        accessions,
                        dna_tensors,
                        [c_id, month],
                        np,
                        prediction_batch_size,
                    )
                    top_predictions = sorted(
                        predictions_by_accession.items(),
                        key=lambda item: item[1],
                        reverse=True,
                    )[:top_n]
                    for accession, probability in top_predictions:
                        forecast_results.append(
                            {
                                "Year": year,
                                "Month": month,
                                "Country": country,
                                "Accession": accession,
                                "Probability": float(probability),
                            }
                        )
                    completed_steps += 1
                    pct = progress_start + (
                        completed_steps / total_steps * (progress_end - progress_start)
                    )
                    _log(progress_callback, f"Scanned {country} {year}-{month:02d}", pct)
                    if scan_bar is not None:
                        scan_bar.update(1)
            _log(progress_callback, f"Completed scan for {country}.")
    finally:
        if scan_bar is not None:
            scan_bar.close()

    results_df = pd.DataFrame(forecast_results)
    if results_df.empty:
        return results_df
    titles = metadata_df[["Accession", "GenBank_Title"]].drop_duplicates("Accession")
    results_df = results_df.merge(titles, on="Accession", how="left")
    return annotate_new_dominant_dna(results_df)


def annotate_new_dominant_dna(results_df):
    """Mark dominant predictions that differ from the previous dominant DNA."""
    if results_df.empty:
        return results_df

    sort_cols = ["Country", "Year", "Month", "Probability"]
    dominant = (
        results_df.sort_values(sort_cols, ascending=[True, True, True, False])
        .drop_duplicates(["Country", "Year", "Month"])
        .sort_values(["Country", "Year", "Month"])
        .copy()
    )
    dominant["Previous_Dominant_Accession"] = dominant.groupby("Country")[
        "Accession"
    ].shift(1)
    if "GenBank_Title" in dominant.columns:
        dominant["Previous_Dominant_Title"] = dominant.groupby("Country")[
            "GenBank_Title"
        ].shift(1)
    else:
        dominant["Previous_Dominant_Title"] = None
    dominant["New_DNA_From_Previous_Dominant"] = (
        dominant["Previous_Dominant_Accession"].notna()
        & (dominant["Accession"] != dominant["Previous_Dominant_Accession"])
    )
    dominant_notes = dominant[
        [
            "Country",
            "Year",
            "Month",
            "Accession",
            "Previous_Dominant_Accession",
            "Previous_Dominant_Title",
            "New_DNA_From_Previous_Dominant",
        ]
    ].rename(columns={"Accession": "Dominant_Accession"})

    annotated = results_df.merge(
        dominant_notes,
        on=["Country", "Year", "Month"],
        how="left",
    )
    annotated["Is_Dominant_Prediction"] = (
        annotated["Accession"] == annotated["Dominant_Accession"]
    )
    annotated["New_DNA_From_Previous_Dominant"] = (
        annotated["Is_Dominant_Prediction"]
        & annotated["New_DNA_From_Previous_Dominant"].fillna(False)
    )
    return annotated


def run_dominant_strain_backtest(
    model,
    dna_tensors,
    country_le,
    metadata_df,
    target_year,
    countries=None,
    deps=None,
    progress_callback=None,
    prediction_batch_size=256,
):
    """Predict the dominant strain for observed country/months in target_year."""
    deps = deps or _load_runtime_deps()
    np = deps["np"]
    pd = deps["pd"]

    test_df = metadata_df[metadata_df["Year"] == target_year].copy()
    if countries:
        test_df = test_df[test_df["Country"].isin(countries)]
    if test_df.empty:
        return pd.DataFrame(), "Dominant backtest unavailable: no rows for target year."

    actual_counts = (
        test_df.groupby(["Country", "Month", "Master_Accession"])
        .size()
        .reset_index(name="Actual_Count")
        .sort_values(["Country", "Month", "Actual_Count"], ascending=[True, True, False])
    )
    actual_top = actual_counts.drop_duplicates(["Country", "Month"])
    actual_top = actual_top.sort_values(["Country", "Month"]).copy()
    actual_top["Previous_Actual_Dominant_Accession"] = actual_top.groupby("Country")[
        "Master_Accession"
    ].shift(1)

    accessions = list(dna_tensors.keys())
    rows = []

    for _, actual in _progress(
        actual_top.iterrows(),
        progress_callback,
        total=len(actual_top),
        desc="Dominant backtest",
        unit="month",
    ):
        country = actual["Country"]
        month = int(actual["Month"])
        try:
            c_id = country_le.transform([country])[0]
        except ValueError:
            continue

        predictions_by_accession = predict_for_all_accessions(
            model,
            accessions,
            dna_tensors,
            [c_id, month],
            np,
            prediction_batch_size,
        )
        predicted, probability = max(
            predictions_by_accession.items(), key=lambda item: item[1]
        )
        actual_accession = actual["Master_Accession"]
        previous_actual = actual.get("Previous_Actual_Dominant_Accession")
        rows.append(
            {
                "Year": int(target_year),
                "Month": month,
                "Country": country,
                "Predicted_Accession": predicted,
                "Predicted_Probability": float(probability),
                "Actual_Accession": actual_accession,
                "Previous_Actual_Dominant_Accession": previous_actual,
                "Actual_Count": int(actual["Actual_Count"]),
                "Actual_New_DNA_From_Previous_Dominant": (
                    pd.notna(previous_actual) and actual_accession != previous_actual
                ),
                "Match": predicted == actual_accession,
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df, "Dominant backtest unavailable: no comparable countries/months."

    result_df = result_df.sort_values(["Country", "Year", "Month"]).copy()
    result_df["Previous_Predicted_Dominant_Accession"] = result_df.groupby("Country")[
        "Predicted_Accession"
    ].shift(1)
    result_df["Predicted_New_DNA_From_Previous_Dominant"] = (
        result_df["Previous_Predicted_Dominant_Accession"].notna()
        & (
            result_df["Predicted_Accession"]
            != result_df["Previous_Predicted_Dominant_Accession"]
        )
    )

    titles = metadata_df[["Accession", "GenBank_Title"]].drop_duplicates("Accession")
    result_df = result_df.merge(
        titles.rename(
            columns={
                "Accession": "Predicted_Accession",
                "GenBank_Title": "Predicted_Title",
            }
        ),
        on="Predicted_Accession",
        how="left",
    )
    result_df = result_df.merge(
        titles.rename(
            columns={
                "Accession": "Actual_Accession",
                "GenBank_Title": "Actual_Title",
            }
        ),
        on="Actual_Accession",
        how="left",
    )
    accuracy = result_df["Match"].mean()
    summary = (
        f"Dominant strain backtest {target_year}: "
        f"{accuracy:.1%} exact-match accuracy across {len(result_df):,} country/months."
    )
    return result_df, summary


def parse_csv_list(value):
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_years(value):
    years = []
    for part in parse_csv_list(value):
        if "-" in part:
            start, end = part.split("-", 1)
            years.extend(range(int(start), int(end) + 1))
        else:
            years.append(int(part))
    return sorted(set(years))


def run_strain_growth_forecast(
    metadata_csv,
    duplicates_json,
    output_csv,
    countries=None,
    years=None,
    training_years=None,
    backtest_year=2025,
    dominant_year=None,
    epochs=10,
    batch_size=256,
    top_n=3,
    progress_callback=None,
):
    deps = _load_runtime_deps()
    np = deps["np"]
    classification_report = deps["classification_report"]
    r2_score = deps["r2_score"]
    Callback = deps["Callback"]

    full_df, dna_tensors, country_le = build_master_dataset(
        metadata_csv, duplicates_json, progress_callback=progress_callback, deps=deps
    )
    _log(progress_callback, "Prepared training dataset.", 10)

    target_year = dominant_year or backtest_year
    if training_years:
        train_data = full_df[
            full_df["Year"].isin(training_years) & (full_df["Year"] < target_year)
        ]
        _log(
            progress_callback,
            f"Training year filter: {min(training_years)}-{max(training_years)} "
            f"(excluding target year {target_year}).",
            12,
        )
    else:
        train_data = full_df[full_df["Year"] < target_year]
        _log(progress_callback, f"Training on rows before {target_year}.", 12)
    test_data = full_df[full_df["Year"] == target_year]
    if train_data.empty:
        raise ValueError(f"No training rows exist before target year {target_year}.")

    X_dna_tr, X_meta_tr, y_tr = prepare_gpu_inputs(
        train_data, dna_tensors, np_module=np, progress_callback=progress_callback
    )
    X_dna_te, X_meta_te, y_te = prepare_gpu_inputs(
        test_data, dna_tensors, np_module=np, progress_callback=progress_callback
    )
    if len(y_tr) == 0:
        raise ValueError("No training examples matched DNA tensors.")

    model = build_multi_input_cnn(deps=deps)
    _log(progress_callback, f"Training model for {epochs} epoch(s)...", 15)
    training_bar = None
    if progress_callback is None and tqdm is not None:
        training_bar = tqdm(total=epochs, desc="Training epochs", unit="epoch")

    class ProgressCallback(Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            pct = 15 + ((epoch + 1) / epochs * 55)
            parts = [
                f"Epoch {epoch + 1}/{epochs}",
                f"loss={logs.get('loss', 0):.4f}",
                f"accuracy={logs.get('accuracy', 0):.4f}",
            ]
            if "val_loss" in logs:
                parts.append(f"val_loss={logs['val_loss']:.4f}")
            if "val_accuracy" in logs:
                parts.append(f"val_accuracy={logs['val_accuracy']:.4f}")
            _log(progress_callback, " | ".join(parts), pct)
            if training_bar is not None:
                training_bar.set_postfix(
                    loss=f"{logs.get('loss', 0):.4f}",
                    accuracy=f"{logs.get('accuracy', 0):.4f}",
                )
                training_bar.update(1)

    validation_split = 0.1 if len(y_tr) >= 10 else 0.0
    try:
        model.fit(
            [X_dna_tr, X_meta_tr],
            y_tr,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            validation_split=validation_split,
            callbacks=[ProgressCallback()],
        )
    finally:
        if training_bar is not None:
            training_bar.close()

    backtest_report = "No backtest rows available."
    r2_summary = "R2 not available: no backtest rows."
    if len(y_te) > 0:
        probabilities = model.predict([X_dna_te, X_meta_te], verbose=0).flatten()
        preds = (probabilities > 0.5).astype(int)
        backtest_report = classification_report(
            y_te, preds, target_names=["Normal", "Outbreak"], zero_division=0
        )
        r2_value = r2_score(y_te, probabilities)
        r2_summary = f"Backtest probability R2: {r2_value:.4f}"
        _log(progress_callback, r2_summary, 72)
        _log(progress_callback, f"Backtest results for {target_year}:\n{backtest_report}", 74)
    else:
        _log(progress_callback, backtest_report, 74)

    dominant_df, dominant_summary = run_dominant_strain_backtest(
        model,
        dna_tensors,
        country_le,
        full_df,
        target_year,
        countries=countries,
        deps=deps,
        progress_callback=progress_callback,
        prediction_batch_size=batch_size,
    )
    dominant_path = None
    if not dominant_df.empty:
        dominant_path = str(
            Path(output_csv).with_name(
                f"{Path(output_csv).stem}_dominant_backtest_{target_year}.csv"
            )
        )
        dominant_df.to_csv(dominant_path, index=False)
    _log(progress_callback, dominant_summary, 74)

    report = run_global_forecast(
        model,
        dna_tensors,
        country_le,
        full_df,
        target_countries=countries,
        years=years,
        top_n=top_n,
        prediction_batch_size=batch_size,
        progress_callback=progress_callback,
        deps=deps,
        progress_start=75,
        progress_end=95,
    )
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    new_forecast_dna_count = 0
    if "New_DNA_From_Previous_Dominant" in report.columns:
        new_forecast_dna_count = int(report["New_DNA_From_Previous_Dominant"].sum())
    new_backtest_dna_count = 0
    if not dominant_df.empty and "Predicted_New_DNA_From_Previous_Dominant" in dominant_df:
        new_backtest_dna_count = int(
            dominant_df["Predicted_New_DNA_From_Previous_Dominant"].sum()
        )
    _log(progress_callback, f"Forecast saved to {output_path}", 100)

    return {
        "output_csv": str(output_path),
        "rows": int(len(report)),
        "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)),
        "backtest_report": backtest_report,
        "r2_summary": r2_summary,
        "target_year": int(target_year),
        "training_years": training_years,
        "dominant_summary": dominant_summary,
        "dominant_backtest_csv": dominant_path,
        "new_forecast_dna_count": new_forecast_dna_count,
        "new_backtest_dna_count": new_backtest_dna_count,
    }


if __name__ == "__main__":
    run_strain_growth_forecast(
        "influenzaA.csv",
        "influenzaA_duplicates.json",
        "global_outbreak_forecast_2020_2025.csv",
    )
