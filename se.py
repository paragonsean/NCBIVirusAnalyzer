

# Run the combination

import pandas as pd
import torch
import gc
import os
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import r2_score # IMPORTANT: Added for the R^2 metric

# --- STEP 1: Highly Optimized DPGR Scoring ---
def prepare_universal_dataset(df):
    print("Preparing 1.78M sequences for Universal DPGR Scoring...")

    # 1. Clean Dates (Operate directly on the dataframe to save memory)
    df['Collection_Date'] = pd.to_datetime(df['Collection_Date'], errors='coerce')
    df.dropna(subset=['Collection_Date', 'sequence', 'Country'], inplace=True)
    df['YearMonth'] = df['Collection_Date'].dt.to_period('M')

    # 2. Extract Lineage/Subtype (Vectorized)
    df['Variant'] = 'Unknown'
    is_b = df['Virus_Type'] == 'Influenza B'
    is_a = df['Virus_Type'] == 'Influenza A'

    # Use loc to quickly assign variants without row-by-row loops
    df.loc[is_b, 'Variant'] = df.loc[is_b, 'GenBank_Title'].str.extract(r'(Victoria|Yamagata)', expand=False).fillna('B/Other')
    df.loc[is_a, 'Variant'] = df.loc[is_a, 'Organism_Name'].str.extract(r'\((H\dN\d)\)', expand=False).fillna('A/Other')

    # 3. Calculate DPGR
    counts = df.groupby(['Country', 'YearMonth', 'Variant']).size().unstack(fill_value=0)
    freqs = counts.div(counts.sum(axis=1), axis=0)
    dpgr = freqs.groupby('Country').diff().shift(-1).fillna(0)

    # 4. VECTORIZED LABEL ASSIGNMENT (The massive speed/RAM fix)
    dpgr_long = dpgr.reset_index().melt(id_vars=['Country', 'YearMonth'], var_name='Variant', value_name='labels')

    df = df.merge(dpgr_long, on=['Country', 'YearMonth', 'Variant'], how='left')
    df['labels'] = df['labels'].fillna(0.0)

    del counts, freqs, dpgr, dpgr_long
    gc.collect()

    # 5. Format Input for Transformer
    df['text'] = "[" + df['Virus_Type'].astype(str) + "][" + df['Country'].astype(str) + "][" + df['Variant'].astype(str) + "] " + df['sequence']

    final_df = df[['text', 'labels']].dropna()

    print(f"Dataset prepared with {len(final_df)} sequences. Saving to disk...")

    # 6. PARQUET DISK BACKING
    parquet_path = "universal_training_data.parquet"
    final_df.to_parquet(parquet_path, index=False)

    del df
    del final_df
    gc.collect()

    return parquet_path

# --- STEP 2: Disk-Streaming Training Pipeline ---
def train_universal_model(parquet_file_path):
    model_name = "zhihan1996/DNA_bert_6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading Dataset via Memory-Mapping...")
    dataset = load_dataset("parquet", data_files=parquet_file_path, split="train")
    dataset = dataset.train_test_split(test_size=0.02)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    print("Tokenizing... (Using multi-processing)")
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

    # --- NEW: R^2 Score Calculation ---
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        # Handle tuple output from model if necessary
        if isinstance(logits, tuple):
            logits = logits[0]
        # Calculate R^2 using sklearn
        r2 = r2_score(labels, logits.squeeze())
        return {"r_squared": r2}

    training_args = TrainingArguments(
        output_dir='./universal_flu_forecaster',
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        num_train_epochs=2,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        logging_strategy="steps",
        logging_steps=1000,           # ALIGNED: Print log every 1000 steps
        eval_strategy="steps",
        eval_steps=1000,              # ALIGNED: Run eval every 1000 steps
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=2,
        load_best_model_at_end=True,  # ALIGNED: Save the best model
        metric_for_best_model="r_squared", # ALIGNED: Use R^2 to pick the best model
        fp16=True,
        report_to="none",
        disable_tqdm=False            # Ensure progress bar is shown
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics # Attach the metrics function
    )

    last_checkpoint = None
    if os.path.isdir('./universal_flu_forecaster'):
        checkpoints = [os.path.join('./universal_flu_forecaster', d) for d in os.listdir('./universal_flu_forecaster') if d.startswith("checkpoint")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"Resuming from checkpoint: {last_checkpoint}")

    print("Commencing Universal Training on 1.78 Million sequences...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

# --- EXECUTION ---
parquet_path = "universal_training_data.parquet"
train_universal_model(parquet_path)