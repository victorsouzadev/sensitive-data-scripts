import os
import pandas as pd

# 📂 Defina a pasta onde estão os arquivos CSV
pasta_csv = os.path.join(os.path.dirname(__file__),'..','data','data-chatgpt')

# 🔍 Listar todos os arquivos que começam com "IOB_relatos_chunk_"
arquivos_csv = sorted([
    os.path.join(pasta_csv, f) for f in os.listdir(pasta_csv) if f.startswith("api_openia_relatos_chunk_") and f.endswith(".csv")
])

print("\n📌 Arquivos encontrados para união:")
print(arquivos_csv)

# 🏗️ Criar lista para armazenar os DataFrames
dataframes = []

# 📚 Ler e adicionar cada arquivo à lista
for arquivo in arquivos_csv:
    df = pd.read_csv(arquivo, encoding="utf-8")
    dataframes.append(df)

# 🔄 Concatenar todos os DataFrames
df_consolidado = pd.concat(dataframes, ignore_index=True)

# 📁 Nome do arquivo de saída
arquivo_saida = os.path.join(pasta_csv, "IOB_relatos_consolidado_chatgpt.csv")

# 💾 Salvar o arquivo consolidado
df_consolidado.to_csv(arquivo_saida, index=False, encoding="utf-8")

print(f"\n✅ Arquivo consolidado salvo em: {arquivo_saida}")
