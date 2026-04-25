import pandas as pd
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, Concatenate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

MAX_SEQ_LENGTH = 2500 

def one_hot_encode_dna(sequence, max_len=MAX_SEQ_LENGTH):
    mapping = {'A': [1,0,0,0], 'C': [0,1,0,0], 'G': [0,0,1,0], 'T': [0,0,0,1], 'N': [0,0,0,0]}
    encoded = np.zeros((max_len, 4), dtype=np.float32)
    for i, char in enumerate(sequence[:max_len]):
        if char in mapping: encoded[i] = mapping[char]
    return encoded

def build_master_dataset(csv_path, json_path):
    print("Loading and mapping sequences...")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    accession_to_master, dna_tensors = {}, {}
    for group in data.get('sequences', []):
        idents = group.get('identifiers', [])
        if not idents: continue
        master = idents[0]
        for ident in idents: accession_to_master[ident] = master
        dna_tensors[master] = one_hot_encode_dna(group.get('sequence', '').upper())

    df = pd.read_csv(csv_path, low_memory=False)
    df['Master_Accession'] = df['Accession'].map(accession_to_master)
    df = df.dropna(subset=['Master_Accession', 'Collection_Date', 'Country'])
    df['Collection_Date'] = pd.to_datetime(df['Collection_Date'], errors='coerce')
    df['YearMonth'] = df['Collection_Date'].dt.to_period('M')
    
    # 1. Target Calculation: Will this strain hit > 5% next month?
    math_df = df.dropna(subset=['YearMonth']).copy()
    monthly_totals = math_df.groupby(['Country', 'YearMonth']).size().reset_index(name='Total')
    strain_monthly = math_df.groupby(['Country', 'YearMonth', 'Master_Accession']).size().reset_index(name='Count')
    growth_df = strain_monthly.merge(monthly_totals, on=['Country', 'YearMonth'])
    growth_df['Freq'] = growth_df['Count'] / growth_df['Total']
    
    growth_df = growth_df.sort_values(by=['Country', 'Master_Accession', 'YearMonth'])
    growth_df['Next_Month_Freq'] = growth_df.groupby(['Country', 'Master_Accession'])['Freq'].shift(-1)
    growth_df['Outbreak'] = (growth_df['Next_Month_Freq'] >= 0.05).astype(int)

    # 2. Metadata Feature Engineering
    final_df = df.merge(growth_df[['Country', 'YearMonth', 'Master_Accession', 'Outbreak']], 
                        on=['Country', 'YearMonth', 'Master_Accession'], how='left').dropna(subset=['Outbreak'])
    
    country_le = LabelEncoder()
    final_df['Country_ID'] = country_le.fit_transform(final_df['Country'])
    final_df['Month'] = final_df['Collection_Date'].dt.month
    
    return final_df, dna_tensors, country_le

def prepare_gpu_inputs(df, dna_tensors):
    X_dna, X_meta, y = [], [], []
    for _, row in df.iterrows():
        master = row['Master_Accession']
        if master in dna_tensors:
            X_dna.append(dna_tensors[master])
            X_meta.append([row['Country_ID'], row['Month']])
            y.append(row['Outbreak'])
    return np.array(X_dna), np.array(X_meta), np.array(y)

def build_multi_input_cnn(num_countries):
    # Branch 1: Sequence Processing
    dna_in = Input(shape=(MAX_SEQ_LENGTH, 4), name="DNA_Input")
    x = Conv1D(32, 4, activation='relu')(dna_in)
    x = MaxPooling1D(2)(x)
    x = Conv1D(64, 8, activation='relu')(x)
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    
    # Branch 2: Regional Context
    meta_in = Input(shape=(2,), name="Meta_Input")
    m = Dense(16, activation='relu')(meta_in)
    
    # Fusion and Decision
    combined = Concatenate()([x, m])
    z = Dense(64, activation='relu')(combined)
    z = Dropout(0.5)(z)
    out = Dense(1, activation='sigmoid')(z)
    
    model = Model(inputs=[dna_in, meta_in], outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

if __name__ == "__main__":
    # --- PHASE 1: DATA SETUP ---
    full_df, dna_tensors, country_le = build_master_dataset("sequences.csv", "sequences_duplicates.json")
    
    # Chronological Split (January 2025 Cutoff)
    cutoff = pd.Period('2025-01', freq='M')
    train_data = full_df[full_df['YearMonth'] < cutoff]
    test_data = full_df[full_df['YearMonth'] >= cutoff]
    
    print(f"\nTraining on Past (Before {cutoff}): {len(train_data)} samples")
    print(f"Testing on Future ({cutoff} onwards): {len(test_data)} samples")

    X_dna_train, X_meta_train, y_train = prepare_gpu_inputs(train_data, dna_tensors)
    X_dna_test, X_meta_test, y_test = prepare_gpu_inputs(test_data, dna_tensors)

    # --- PHASE 2: TRAINING ---
    model = build_multi_input_cnn(len(country_le.classes_))
    print("\nIgniting RTX 3090 Multi-Input Training...")
    
    model.fit(
        [X_dna_train, X_meta_train], y_train,
        validation_data=([X_dna_test, X_meta_test], y_test),
        epochs=15, batch_size=64
    )
    
    # --- PHASE 3: INTERACTIVE INFERENCE ---
    def predict_outbreak(accession, country_name, month):
        if accession not in dna_tensors: return "DNA not found"
        c_id = country_le.transform([country_name])[0]
        prob = model.predict([dna_tensors[accession][np.newaxis, ...], np.array([[c_id, month]])])[0][0]
        return f"Probability of Outbreak in {country_name} (Month {month}): {prob:.2%}"

    print("\n--- Model is ready for prediction ---")
    # Example usage:
    print(predict_outbreak('OP546698.1', 'France', 2))