import json
import pandas as pd
from collections import defaultdict
import os

def iob_to_json(iob_data, doc_id):
    entities = []
    entity = None
    entity_id_counter = 1

    # Verificar se as colunas necessárias existem
    expected_columns = {"row_id", "Palavra", "Tag IOB"}
    missing_columns = expected_columns - set(iob_data.columns)
    if missing_columns:
        raise KeyError(f"Colunas ausentes no CSV: {missing_columns}")

    # Substituir NaN por string vazia ou "O"
    iob_data["Palavra"] = iob_data["Palavra"].fillna("").astype(str)
    iob_data["Tag IOB"] = iob_data["Tag IOB"].fillna("O").astype(str)

    # Agrupar sentenças pelo row_id
    grouped_sentences = defaultdict(list)
    sentence_offsets = {}
    start_offsets = {}
    current_offset = 0
    
    for _, row in iob_data.iterrows():
        row_id = row["row_id"]
        token = row["Palavra"]
        if row_id not in grouped_sentences:
            start_offsets[row_id] = 0
            sentence_offsets[row_id] = current_offset
        grouped_sentences[row_id].append(token)
        current_offset += len(token) + 1  # Espaço entre palavras
    
    results = []
    for row_id, tokens in grouped_sentences.items():
        doc_text = " ".join(tokens)
        sentence_entities = []
        start_offset = 0  # Agora os offsets são relativos ao doc_text da sentença
        entity = None
        entity_id_counter = 1
        
        for _, row in iob_data[iob_data["row_id"] == row_id].iterrows():
            token, tag = row["Palavra"], row["Tag IOB"]
            token_start = start_offset
            token_end = token_start + len(token)
            
            if tag.startswith("B-"):
                if entity:
                    sentence_entities.append(entity)
                entity = {
                    "entity_id": f"{doc_id}-{row_id}-{entity_id_counter}",
                    "text": token,
                    "label": tag[2:],
                    "start_offset": token_start,
                    "end_offset": token_end
                }
                entity_id_counter += 1
            elif tag.startswith("I-") and entity and entity["label"] == tag[2:]:
                entity["text"] += " " + token
                entity["end_offset"] = token_end
            else:
                if entity:
                    sentence_entities.append(entity)
                    entity = None
            
            start_offset = token_end + 1  # Espaço entre palavras

        if entity:
            sentence_entities.append(entity)
        
        # Filtrar sentenças sem anotações
        if sentence_entities:
            results.append({
                "doc_id": f"{doc_id}-{row_id}",
                "doc_text": doc_text,
                "entities": sentence_entities
            })
    
    return results

# Leitura do arquivo CSV e execução do script
def read_iob_csv(file_path):
    df = pd.read_csv(file_path, delimiter=',')  # Adapte o delimitador conforme necessário
    print("Colunas encontradas:", df.columns)  # Debugging
    return df

# Exemplo de uso
iob_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'data-chatgpt', 'IOB_relatos_consolidado_chatgpt_clear.csv')

iob_data = read_iob_csv(iob_file)
doc_id = "H2-dftre765"

output_json = iob_to_json(iob_data, doc_id)

# Salvando em um arquivo JSON
with open("output_chatgpt.json", "w", encoding="utf-8") as f:
    json.dump(output_json, f, ensure_ascii=False, indent=4)

print("Conversão concluída. JSON salvo como output_chatgpt.json")
