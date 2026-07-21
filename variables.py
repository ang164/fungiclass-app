# Percorsi dei principali file che verranno utilizzati
FASTA_FILE = "dataset/CBSITS.fasta"
CLASS_FILE = "dataset/CBSITS.current.classification"

FASTA_FILEITS= "newdataset/2025_12_CBS_ITS/2025_12_CBS_ITS.fasta_reconstructed.fasta"
FASTA_FILE26S= "newdataset/2025_12_CBS_26S/2025_12_CBS_26S.fasta_reconstructed.fasta"

file_crypto= "dataset/cryptococcus.txt"
file_cerevisiae="dataset/scerevisiae.txt"
file_candida= "dataset/Candidaorthopsilosis.txt"
file_gatti="dataset/cryptogatti.txt"
file_pastorianus="dataset/pastorianus.txt"

# Parametri del modello
K_MER_SIZE = 6
N_SPECIES = 10
#CONFIDENCE_THRESHOLD = 0.8
THRESHOLDS = {
    "phylum": 0.84,
    "class":  0.68,
    "order":  0.72,
    "family": 0.64,
    "genus":  0.84,
    "species": 0.84,
}
# Mappa IUPAC modificata per "azzerare" le ambiguità
# Tutte le lettere ambigue diventano '0'
IUPAC_ZERO_MAP = {
    "A": "A",
    "C": "C",
    "G": "G",
    "T": "T",
    "R": "0",
    "Y": "0",
    "S": "0",
    "W": "0",
    "K": "0",
    "M": "0",
    "B": "0",
    "D": "0",
    "H": "0",
    "V": "0",
    "N": "0"
}

# --- PARAMETRI XGBOOST ---
# Dizionario di configurazione per il classificatore
XGB_PARAMS_CV_A = {
    "n_estimators": 200,       # Numero massimo di alberi
    "learning_rate": 0.1,      # Velocità di apprendimento (più basso = più robusto ma lento)
    "max_depth": 7,             # Profondità massima degli alberi (evita overfitting)
    "subsample": 0.8,           # Frazione di dati usata per ogni albero
    "colsample_bytree": 0.8,    # Frazione di feature (k-mers) usata per albero
    "objective": "multi:softprob", # Output: probabilità per ogni classe
    "n_jobs": -1,               # Usa tutti i core della CPU
    "random_state": 7,          # Riproducibilità
    "eval_metric": "mlogloss",  # Metrica di errore per multiclasse
    "tree_method": "hist"       # Metodo istogramma (molto veloce per grandi dataset)
}
XGB_PARAMS_CV_B = {
    "n_estimators": 180,       # Numero massimo di alberi
    "learning_rate": 0.1,      # Velocità di apprendimento (più basso = più robusto ma lento)
    "max_depth": 6,             # Profondità massima degli alberi (evita overfitting)
    "subsample": 0.8,           # Frazione di dati usata per ogni albero
    "colsample_bytree": 0.8,    # Frazione di feature (k-mers) usata per albero
    "objective": "multi:softprob", # Output: probabilità per ogni classe
    "n_jobs": -1,               # Usa tutti i core della CPU
    "random_state": 7,          # Riproducibilità
    "eval_metric": "mlogloss",  # Metrica di errore per multiclasse
    "tree_method": "hist"       # Metodo istogramma (molto veloce per grandi dataset)
}
XGB_PARAMS_FINAL = {
    "n_estimators":100,       # Numero massimo di alberi
    "learning_rate": 0.05,      # Velocità di apprendimento (più basso = più robusto ma lento)
    "max_depth": 7,             # Profondità massima degli alberi (evita overfitting)
    "subsample": 0.8,           # Frazione di dati usata per ogni albero
    "colsample_bytree": 0.8,    # Frazione di feature (k-mers) usata per albero
    "objective": "multi:softprob", # Output: probabilità per ogni classe
    "n_jobs": -1,               # Usa tutti i core della CPU
    "random_state": 7,          # Riproducibilità
    "eval_metric": "mlogloss",  # Metrica di errore per multiclasse
    "tree_method": "hist"       # Metodo istogramma (molto veloce per grandi dataset)
}

LEVEL_PARAMS = {
    "phylum": {"n_estimators": 200, "max_depth": 8},
    "class": {"n_estimators": 150, "max_depth": 7},
    "order": {"n_estimators": 120, "max_depth": 6},
    "family": {"n_estimators": 100, "max_depth": 6},
    "genus": {"n_estimators": 80, "max_depth": 5, "learning_rate": 0.1},
    "species": {"n_estimators": 60, "max_depth": 4, "learning_rate": 0.1}
}

