import pandas as pd
import os

# Definir o nome do arquivo
arquivo_csv =os.path.join(os.path.dirname(__file__), "relatos.csv")

# Tamanho do chunk
chunk_size = 1000

# Criar um gerador de chunks
chunks = pd.read_csv(arquivo_csv, chunksize=chunk_size)

# Processar cada chunk
for i, chunk in enumerate(chunks):
    # Nome do arquivo de saída
    output_file = os.path.join(os.path.dirname(__file__), f'split/relatos_chunk_{i + 1}.csv') 
    
    # Salvar o chunk
    chunk.to_csv(output_file, index=False)
    
    print(f'Chunk {i + 1} salvo como {output_file}')
