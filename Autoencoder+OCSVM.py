import numpy as np
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score
import dataset
# ==========================================
# DATA PREPARATION 
# ==========================================
num_features = 41
X_train_raw = dataset.getTrainDataSet()
# labels: 0 for Normal, 1 for Attack (e.g., neptune, satan)
y_train_raw = X_train_raw[["class_label"]]


X_test_raw = dataset.getTestDataSet()
y_test_raw = X_test_raw[["class_label"]]


# Record the dataset lengths so we can accurately separate them later
train_len = len(X_train_raw)

# Combine datasets before encoding to align categorical columns perfectly
combined_df = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)
# Apply One-Hot Encoding to the 3 categorical columns
categorical_cols = ['protocol_type', 'service', 'flag']
# Apply log transform to heavily skewed numerical columns
skewed_cols = ['duration', 'src_bytes', 'dst_bytes', 'wrong_fragment', 'urgent', 
               'hot', 'num_failed_logins', 'num_compromised', 'num_root', 
               'num_file_creations', 'num_shells', 'num_access_files']

for col in skewed_cols:
    if col in X_train_raw.columns:
        X_train_raw[col] = np.log1p(X_train_raw[col].astype(float))
    if col in X_test_raw.columns:
        X_test_raw[col] = np.log1p(X_test_raw[col].astype(float))

combined_df_encoded = pd.get_dummies(combined_df, columns=categorical_cols, dtype=int)
#Split back into Train and Test DataFrames
train_chunk = combined_df_encoded.iloc[:train_len]
X_train_raw = combined_df_encoded.iloc[:train_len].drop(columns=["class_label","difficulty_score"])
y_train_raw = train_chunk[['class_label']].copy()

test_chunk = combined_df_encoded.iloc[train_len:]
X_test_raw = combined_df_encoded.iloc[train_len:].drop(columns=["class_label","difficulty_score"])
y_test_raw = test_chunk[['class_label']].copy()
# Scale features between 0 and 1
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# CRITICAL: OCSVM and AE are trained ONLY on "Normal" data (class 0)
y_train_raw["class_label"]=y_train_raw["class_label"].map(lambda x: 0 if x=="normal" else 1)
X_train_normal = X_train_scaled[y_train_raw.squeeze() == 0]

y_test_raw["class_label"]=y_test_raw["class_label"].map(lambda x: 0 if x=="normal" else 1)
#print(X_train_normal)

# ==========================================
# AUTOENCODER ARCHITECTURE (Feature Extractor)
# ==========================================
input_dim = X_train_normal.shape[1]
input_layer = Input(shape=(input_dim,), name="Input_Layer")

# Encoder
encoded = Dense(64, activation='relu', name="Encoder_1")(input_layer)
encoded = Dense(32, activation='relu', name="Encoder_2")(encoded)

# Latent Bottleneck (The extracted features will come from here)
latent_space = Dense(16, activation='relu', name="Latent_Bottleneck")(encoded)

# Decoder
decoded = Dense(32, activation='relu', name="Decoder_1")(latent_space)
decoded = Dense(64, activation='relu', name="Decoder_2")(decoded)
output_layer = Dense(input_dim, activation='sigmoid', name="Output_Layer")(decoded)

# Instantiate the full Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=output_layer)
autoencoder.compile(optimizer='adam', loss='mse')

# Train Autoencoder on Normal traffic to minimize reconstruction error
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=20,
    batch_size=250,
    shuffle=True,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# EXTRACT LATENT FEATURES
# ==========================================
# Define a sub-model that isolates the output of the bottleneck layer
encoder_model = Model(inputs=input_layer, outputs=latent_space)

# Compress the high-dimensional data into 16-dimension dense features
X_train_features = encoder_model.predict(X_train_normal)
X_test_features = encoder_model.predict(X_test_scaled)

# ==========================================
# ONE-CLASS SVM CLASSIFICATION
# ==========================================
# nu: Upper bound on training errors (approximate % of anomalies in training data)
# gamma: Kernel coefficient for RBF (controls how tightly the boundary wraps)
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
oc_svm.fit(X_train_features)

# Predict anomalies on test data
# scikit-learn OneClassSVM outputs: 1 for inliers (Normal), -1 for outliers (Attack)
svm_preds = oc_svm.predict(X_test_features)

# Map predictions to match our labels: Normal -> 0, Attack -> 1
y_pred = np.where(svm_preds == 1, 0, 1)

# ==========================================
# EVALUATION
# ==========================================
print("\n=== Model Performance ===")
print("Accuracy:", accuracy_score(y_test_raw, y_pred))
print("\nClassification Report:")
print(classification_report(y_test_raw, y_pred, target_names=['Normal', 'Attack']))
