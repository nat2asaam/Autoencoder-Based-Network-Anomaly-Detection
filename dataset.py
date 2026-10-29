import pandas as pd

# Define the full list of 43 column names (41 features + label + difficulty score)
column_names = [
    # 1. Basic features
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 
    'land', 'wrong_fragment', 'urgent',
    # 2. Content features
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    # 3. Time-based traffic features
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    # 4. Host-based traffic features
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    # Target variables
    'class_label', 'difficulty_score'
]

# Load the dataset (replace with your local file path)
# Note: header=None prevents pandas from mistakenly treating row 0 as a header
df_train = pd.read_csv('data/KDDTrain+.txt/KDDTrain+.txt', names=column_names, header=None)
df_test =pd.read_csv('data/KDDTest+.txt/KDDTest+.txt', names=column_names, header=None)

def viewTrainDataSet():
    # Quick sanity check
    print("Train Dataset Shape:", df_train.shape)
    print("\nFirst 5 rows:")
    print(df_train.head())
def getTrainDataSet():
    return df_train
def viewTestDataSet():
    # Quick sanity check
    print("Train Dataset Shape:", df_test.shape)
    print("\nFirst 5 rows:")
    print(df_test.head())
def getTestDataSet():
    return df_test


