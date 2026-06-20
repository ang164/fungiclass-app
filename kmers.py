import pandas as pd

from variables import IUPAC_ZERO_MAP

def clean_iupac_sequence(seq):
    """
    Pulisce la sequenza di DNA.
    Converte le basi standard (ACGT) e trasforma le basi ambigue (es. N, R, Y) in '0'
    utilizzando la mappa definita in variables.py.
    """
    # List comprehension per velocità: mappa ogni carattere, default '0' se sconosciuto
    return "".join([IUPAC_ZERO_MAP.get(base, "0") for base in seq])

def get_clean_kmers_string(seq, k=6):
    """
    Genera una stringa di k-mers separati da spazi (es. "ATCGGG TCGGGA ...").
    Logica: Scarta i k-mer che contengono '0' (ambiguità).
    Input: Sequenza DNA stringa.
    Output: Stringa di testo pronta per TF-IDF.
    """
    seq = clean_iupac_sequence(seq)
    kmers = []
    # Finestra scorrevole sulla sequenza
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        # Se il k-mer è "pulito" (solo ACGT), lo teniamo
        if '0' not in kmer:
            kmers.append(kmer)
            
    return " ".join(kmers)

def prepare_dataset(df, k=6):
    """
    Funzione principale da chiamare sul DataFrame.
    Crea la colonna 'kmers_text' necessaria per il training.
    """
    print("Generazione K-mers in corso...")
    # Applica la trasformazione a ogni riga della colonna 'sequence' e 
    # crea una nuova colonna con i k-mers pronti per il TfidfVectorizer 
    df['kmers_text'] = df['sequence'].apply(lambda x: get_clean_kmers_string(x, k))
    
    # Rimuove righe che non hanno prodotto k-mer validi (es. sequenze di sole 'N')
    df = df[df['kmers_text'].str.len() > 0].copy()
    return df

###############################################################

def get_clean_kmers_string_0(seq, k=6):
    
    seq = clean_iupac_sequence(seq)
    kmers = []
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i+k]
        # Se vuoi che il modello ignori totalmente l'ambiguità,
        # scartiamo i kmer che contengono '0'
        if '0' in kmer:
            kmer = '0' * k
        kmers.append(kmer)
            
    return " ".join(kmers)

def prepare_dataset_0(df, k=6):
    """Applica la generazione dei k-mer a tutto il dataframe."""
    print("Generazione K-mers in corso...")
    # Crea una nuova colonna con i k-mers pronti per il CountVectorizer
    df['kmers_text'] = df['sequence'].apply(lambda x: get_clean_kmers_string_0(x, k))
    
    # Rimuoviamo sequenze che sono diventate vuote (troppi '0')
    #df = df[df['kmers_text'].str.len() > 0].copy()
    return df
