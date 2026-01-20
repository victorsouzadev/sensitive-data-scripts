import os
import re
import unicodedata
import pandas as pd
import logging

# -------------------
# Configuração de logs
# -------------------
log_file = "processamento_dataset.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

# -------------------
# Arquivos e pastas
# -------------------
file_relatos = 'dataset/data/top_10_consolidado_maraba_full.csv'
path_dataset = 'dataset-classificacao-v2/dataset/'

try:
    # 1) Lê CSV original
    logging.info(f"Lendo dataset de entrada: {file_relatos}")
    data = pd.read_csv(file_relatos)
    logging.info(f"Dataset lido com {len(data)} registros.")

    # 2) Seleciona colunas de interesse
    df_export = data[['relato','consolidado','municipios','distrito','regionais','bairros']]
    os.makedirs(path_dataset, exist_ok=True)

    out_csv = os.path.join(path_dataset, 'temp/top_10_consolidado_maraba_coluna_interesse.csv')
    df_export.to_csv(out_csv, index=False)
    logging.info(f"Arquivo exportado com colunas de interesse: {out_csv}")

    # 3) Relê e remove duplicados
    df_analise = pd.read_csv(out_csv)
    df = df_analise.drop_duplicates(subset=['relato']).copy()
    logging.info(f"Após remover duplicados: {len(df)} registros restantes.")

    # -------------------
    # Funções de limpeza
    # -------------------
    def remove_diacritics_text(text: str) -> str:
        if not isinstance(text, str):
            return ''
        nfkd = unicodedata.normalize("NFKD", text)
        return ''.join(c for c in nfkd if not unicodedata.combining(c))

    def clean_series(s: pd.Series) -> pd.Series:
        s = s.astype('string').fillna('')

        # remove HTML/entidades/sinais repetidos
        s = s.str.replace(
            r'(<\/?\w*(?:\s+style=".*?")?>|&[npsb]+;|/+|\++|\*+|_+)',
            '',
            regex=True
        )
        s = s.str.replace(r'\s+', ' ', regex=True)  # múltiplos espaços
        s = s.str.strip()
        s = s.map(remove_diacritics_text)
        return s

    # 4) Limpeza da coluna 'relato'
    logging.info("Iniciando limpeza da coluna 'relato'...")
    df.loc[:, 'relato'] = clean_series(df['relato'])
    logging.info("Limpeza concluída.")

    # 5) Exporta dataset limpo
    out_dataset_clear = os.path.join(path_dataset, 'top_10_consolidado_maraba_coluna_interesse_clean.csv')
    df.to_csv(out_dataset_clear, index=False)
    logging.info(f"Dataset final exportado para: {out_dataset_clear}")

except Exception as e:
    logging.error(f"Erro durante o processamento: {e}", exc_info=True)
