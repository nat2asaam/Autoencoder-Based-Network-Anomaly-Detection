import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from keras import layers
import dataset
# ==========================================
# NETWORK DATA (NSL_KDD dataset)
# ==========================================
# get dataset
num_features = 41
all_traffic=dataset.getTrainDataSet()
# filter normal traffic
normal_traffic=all_traffic[all_traffic["class_label"]=="normal"]
# filter anomalous network traffic (e.g., spikes in bytes/packets from an attack)
attack_traffic=all_traffic[all_traffic["class_label"]!="normal"]
X_train=normal_traffic

X_test=dataset.getTestDataSet()
y_test=X_test[["class_label"]]
#X_test=X_test.drop(columns=["class_label","difficulty_score"])

# Record the dataset lengths so we can accurately separate them later
train_len = len(X_train)

# Combine datasets before encoding to align categorical columns perfectly
combined_df = pd.concat([X_train, X_test], axis=0, ignore_index=True)
# Apply One-Hot Encoding to the 3 categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
# Apply log transform to heavily skewed numerical columns
skewed_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 
               'hot', 'num_failed_logins', 'num_compromised', 'num_root', 
               'num_file_creations', 'num_shells', 'num_access_files']

for col in skewed_cols:
    if col in all_traffic.columns:
        X_train[col] = np.log1p(X_train[col].astype(float))
    if col in X_test.columns:
        X_test[col] = np.log1p(X_test[col].astype(float))

combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, dtype=int)

# Split back into Train and Test DataFrames
train_chunk = combined_df_encoded.iloc[:train_len]
X_train = combined_df_encoded.iloc[:train_len].drop(columns=["class_label","difficulty_score"])
y_train = train_chunk[['class_label']].copy()

test_chunk = combined_df_encoded.iloc[train_len:]
X_test = combined_df_encoded.iloc[train_len:].drop(columns=["class_label","difficulty_score"])
y_test = test_chunk[['class_label']].copy()

# Remember: Autoencoders must be trained ONLY on normal traffic
print(X_test.head())

# Mix normal and attack data for evaluation
#X_test = np.vstack([X_test_normal, attack_traffic])
# 0 = Normal, 1 = Anomaly
y_test["class_label"]=y_test["class_label"].map(lambda x: 0 if x=="normal" else 1)
# ==========================================
# DATA PREPROCESSING
# ==========================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
#  BUILD THE AUTOENCODER MODEL
# ==========================================
input_dim = X_train_scaled.shape[1]

model = keras.Sequential([
    layers.Input(shape=(input_dim,)),
    
    # Encoder
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'), # Bottleneck layer (latent space)
    
    # Decoder
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.1),
    layers.Dense(64, activation='relu'),
    layers.Dense(input_dim, activation='linear') 
])


model.compile(optimizer='adam', loss='mse', metrics=['accuracy','precision'])
model.summary()

# ==========================================
# TRAIN THE MODEL (On Normal Data Only)
# ==========================================
# Notice that y targets are identical to the inputs (X_train_scaled)
history = model.fit(
    X_train_scaled, X_train_scaled,
    epochs=30,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# ==========================================
# DEFINE ANOMALY THRESHOLD
# ==========================================
# Calculate reconstruction error on the training set
train_predictions = model.predict(X_train_scaled)
train_mae_loss = np.mean(np.abs(train_predictions - X_train_scaled), axis=1)
# 1. Update your training error calculation to use MSE (squaring the differences)
#train_mse_loss = np.mean(np.square(train_predictions - X_train_scaled), axis=1)
# Set threshold at the 95th percentile of normal data error
threshold = np.percentile(train_mae_loss, 85)
print(f"\n[INFO] Anomaly decision threshold set at MSE score: {threshold:.4f}")

# ==========================================
# 6. DETECT ANOMALIES ON TEST TRAFFIC
# ==========================================
test_predictions = model.predict(X_test_scaled)
test_mae_loss = np.mean(np.abs(test_predictions - X_test_scaled), axis=1)

# Flag as anomaly if the reconstruction loss exceeds the threshold
predicted_anomalies = (test_mae_loss > threshold).astype(int)

# Quick evaluation breakdown
actual_anomalies_detected = np.sum((predicted_anomalies == 1) & (y_test['class_label'].values.flatten() == 1))
y_true_test = y_test['class_label'].values.flatten()
total_test_attacks = np.sum(y_true_test == 1)
print(f"[RESULT] Successfully flagged {actual_anomalies_detected}/{total_test_attacks} actual test network attacks.")
print(f"Detection Rate (Recall): {(actual_anomalies_detected / total_test_attacks) * 100:.2f}%")

