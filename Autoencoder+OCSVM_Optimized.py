import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score
import dataset

# ==========================================
# DATA PREPARATION 
# ==========================================
X_train_raw = dataset.getTrainDataSet()
X_test_raw = dataset.getTestDataSet()

train_len = len(X_train_raw)

# Combine datasets first
combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# FIX: Apply log transform directly to the combined DataFrame
skewed_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 
               'hot', 'num_failed_logins', 'num_compromised', 'num_root', 
               'num_file_creations', 'num_shells', 'num_access_files']

for col in skewed_cols:
    if col in combined_df.columns:
        combined_df[col] = np.log1p(combined_df[col].astype(float))

# One-Hot Encoding
categorical_cols = ['protocol_type', 'service', 'flag']
combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, dtype=int)

# Split back cleanly
train_chunk = combined_df_encoded.iloc[:train_len]
X_train_raw = train_chunk.drop(columns=["class_label", "difficulty_score"])
y_train_raw = train_chunk[['class_label']].copy()

test_chunk = combined_df_encoded.iloc[train_len:]
X_test_raw = test_chunk.drop(columns=["class_label", "difficulty_score"])
y_test_raw = test_chunk[['class_label']].copy()

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Clean labels and extract Normal data
y_train_raw["class_label"] = y_train_raw["class_label"].map(lambda x: 0 if x == "normal" else 1)
X_train_normal = X_train_scaled[y_train_raw.squeeze() == 0]

y_test_raw["class_label"] = y_test_raw["class_label"].map(lambda x: 0 if x == "normal" else 1)

# ==========================================
# ENHANCED AUTOENCODER ARCHITECTURE
# ==========================================
input_dim = X_train_normal.shape[1]
input_layer = Input(shape=(input_dim,), name="Input_Layer")

# Encoder with Batch Normalization
encoded = Dense(64, activation='relu', name="Encoder_1")(input_layer)
encoded = BatchNormalization()(encoded)
encoded = Dense(32, activation='relu', name="Encoder_2")(encoded)
encoded = BatchNormalization()(encoded)

# Latent Bottleneck
latent_space = Dense(44, activation='relu', name="Latent_Bottleneck")(encoded)

# Decoder
decoded = Dense(32, activation='relu', name="Decoder_1")(latent_space)
decoded = BatchNormalization()(decoded)
decoded = Dense(64, activation='relu', name="Decoder_2")(decoded)
decoded = BatchNormalization()(decoded)
output_layer = Dense(input_dim, activation='sigmoid', name="Output_Layer")(decoded)

autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse',metrics=['accuracy','precision'])

# Callbacks for better training stability
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.5, verbose=1)

autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=40,            # Increased epochs
    batch_size=128,       # Smaller batch size helps gradient descent explore better local minima
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop, lr_reduction],
    verbose=1
)

# ==========================================
# EXTRACT LATENT FEATURES
# ==========================================
encoder_model = Model(inputs=input_layer, outputs=latent_space)
X_train_features = encoder_model.predict(X_train_normal)
X_test_features = encoder_model.predict(X_test_scaled)

# ==========================================
# ONE-CLASS SVM CLASSIFICATION (Optimized)
# ==========================================
# Tuned nu to 0.005 assuming very low anomalies exist in pure normal training data
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.005)
oc_svm.fit(X_train_features)

svm_preds = oc_svm.predict(X_test_features)
y_pred = np.where(svm_preds == 1, 0, 1)

# ==========================================
# EVALUATION
# ==========================================
print("\n=== Model Performance ===")
print("Accuracy:", accuracy_score(y_test_raw, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_raw, y_pred, target_names=['Normal', 'Attack']))
