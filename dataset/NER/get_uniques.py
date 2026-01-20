import pandas as pd
import os
from collections import Counter

# 📂 Defina a pasta onde estão os arquivos CSV
file_path = os.path.join(os.path.dirname(__file__),'..','data','data-chatgpt','IOB_relatos_consolidado_chatgpt.csv')

# Ler o arquivo CSV
df = pd.read_csv(file_path)

# Verificar se as colunas esperadas existem
if "predicted_iob_tag" in df.columns and "token" in df.columns:
    tag_column = "predicted_iob_tag"
    word_column = "token"
else:
    tag_column = "Tag IOB"
    word_column = "Palavra"

# Mapeamento principal (já existente)
mapeamento_tags = {
    # Pessoa
    "B-PESSOA": "B-PESSOA", "I-PESSOA": "I-PESSOA", "D-PESSOA": "B-PESSOA", 
    "E-PESSOA": "I-PESSOA", "B-PESS": "B-PESSOA", "I-PESSOAA": "I-PESSOA",
    "O-PESSOA": "O", "I-PESSOA)": "I-PESSOA", "B-P": "B-PESSOA", 
    "I-PESSOA.": "I-PESSOA", "I-B-PESSOA": "I-PESSOA", "I-I-PESSOA": "I-PESSOA",
    "B-RELATORA": "B-PESSOA", "I-RELATORA": "I-PESSOA",
    "B-RELATOR": "B-PESSOA", "I-RELATOR": "I-PESSOA",
    "B-DEPONENTE": "B-PESSOA", "I-DEPONENTE": "I-PESSOA",
    "B-DECLARANTE": "B-PESSOA", "I-DECLARANTE": "I-PESSOA",

    # Veículo
    "B-VEÍCULO": "B-VEÍCULO", "I-VEÍCULO": "I-VEÍCULO", 
    "B-Veículo": "B-VEÍCULO", "I-Veículo": "I-VEÍCULO",
    "B-VeÍCULO": "B-VEÍCULO", "I-VeÍCULO": "I-VEÍCULO",
    "O-VEÍCULO": "O", "VEÍCULO": "B-VEÍCULO", "C-Veículo": "B-VEÍCULO",

    # Endereço
    "B-ENDEREÇO": "B-ENDEREÇO", "I-ENDEREÇO": "I-ENDEREÇO",
    "B-LOCAL": "B-ENDEREÇO", "I-LOCAL": "I-ENDEREÇO",
    "B-ESTADO": "B-ENDEREÇO", "O-ENDEREÇO": "O",
    "B-SALA": "B-ENDEREÇO", "B-PAVILHÃO": "B-ENDEREÇO",
    "B-RUA": "B-ENDEREÇO", "I-RUA": "I-ENDEREÇO",

    # Documentos
    "B-CPF": "B-CPF", "I-CPF": "I-CPF", "B-CPF,": "B-CPF", "I-CPF,": "I-CPF", "I-CPF)": "I-CPF",
    "B-RG": "B-RG", "I-RG": "I-RG", "I-RG,": "I-RG", "I-RG)": "I-RG",
    "B-CNPJ": "B-CNPJ", "I-CNPJ": "I-CNPJ", "B-CNPJ;": "B-CNPJ",
    "B-CNH": "B-CNH", "I-CNH": "I-CNH",
    "B-REG": "O", "I-REG": "O", "B-REGISTRO": "O", "I-REGISTRO": "O",
    "B-RENAVAM": "O", "I-RENAVAM": "O", "B-CHASSIS": "O",
    "B-CRM": "O", "I-CRM": "O", "B-IDENTIDADE": "B-RG", "I-IDENTIDADE": "I-RG",

    # Organizações
    "B-EMPRESA": "B-EMPRESA", "I-EMPRESA": "I-EMPRESA",
    "B-BANCO": "B-BANCO", "I-BANCO": "I-BANCO",
    "O-EMPRESA": "O", "O-BANCO": "O", "B-EMP": "B-EMPRESA", "B-MEI": "B-EMPRESA",
    "B-EMPRESA.": "B-EMPRESA", "I-EMPRESA.": "I-EMPRESA",
    "IEMPRESA": "I-EMPRESA", "I-EMPRESA;": "I-EMPRESA", "I-EMPRESA”:": "I-EMPRESA",

    # Contato
    "B-TELEFONE": "B-TELEFONE", "I-TELEFONE": "I-TELEFONE",
    "B-EMAIL": "B-EMAIL", "I-EMAIL": "I-EMAIL", "D-EMAIL": "B-EMAIL", "S-EMAIL": "B-EMAIL",
    "I-TELEFONE)": "I-TELEFONE", "I-TELEFONE);": "I-TELEFONE", "I-TELEFONE,": "I-TELEFONE", "I-TELEFONE),": "I-TELEFONE",

    # PIX / CHAVE
    "B-PIX": "B-PIX", "I-PIX": "I-PIX", "B-CHAVE": "B-PIX", "I-CHAVE": "I-PIX", "B-CHAVE_PIX": "B-PIX", "B-Pix": "B-PIX",

    # Correções gerais de lixo
    "O)": "O", "O.": "O", "O,": "O", "O;": "O", "O:": "O", "O\")": "O",
    "O),": "O", "O).": "O", "O\"": "O", "O”": "O", "O”:": "O", "O/:": "O",
    "O?”": "O", "O°.": "O", "O...": "O", "O”;": "O", "O”,": "O", "O”.": "O",
    "O-HORA": "O", "O-EMAIL": "O", "O-DIA": "O", "O-MAE": "O", "O-CPF": "O",
    "O-CNH": "O", "O-NOTICIA": "O", "O-DIRETORA": "O", "O-": "O", "MAS": "O",
    "A": "O", "I": "O", "U": "O", "S": "O", "V": "O", "E": "O"
}

# Aplicar o mapeamento
df[tag_column] = df[tag_column].replace(mapeamento_tags)

mapeamento_tags_extra = {
    # 🔧 Correções e sujeiras diversas
    "&NBSP;": "O", "B-O": "O", "I-O": "O", "I-BE": "O", "IEMPRESA": "I-EMPRESA",
    "I-PESSOA\",": "I-PESSOA", "I-PESSOA\":": "I-PESSOA", "I-PESSOA),": "I-PESSOA",
    "I-PESSOA”,": "I-PESSOA", "I-PESSOA;": "I-PESSOA", "I-PESSOA”:": "I-PESSOA",
    "I-ENDEREÇO),": "I-ENDEREÇO", "I-ENDEREÇO).": "I-ENDEREÇO",
    "I-ENDEREÇO,": "I-ENDEREÇO", "I-ENDEREÇO;": "I-ENDEREÇO", "I-ENDEREÇO.:": "I-ENDEREÇO",
    "B-ENDEREÇO,": "B-ENDEREÇO", "B-ENDEREÇO.:": "B-ENDEREÇO",

    # 📅 Tempo
    "B-DATA": "O", "I-DATA": "O", "B-HORA": "O", "I-HORA": "O", "B-DIA": "O", "I-DIA": "O",
    "B-MÊS": "O", "I-MÊS": "O", "B-ANO": "O", "I-ANO": "O", "B-SEMANA": "O", "I-SEMANA": "O",
    "B-MES": "O", "I-MES": "O", "B-MINUTO": "O", "I-MINUTO": "O",

    # 👤 Pessoa (variações válidas)
    "B-PESSOA,": "B-PESSOA", "I-PESSOA,": "I-PESSOA", "B-PESSOA.": "B-PESSOA",
    "I-PESSOA.": "I-PESSOA", "B-PESSOAA": "B-PESSOA", "I-PESSOAA": "I-PESSOA",
    "B-RPESSOA": "B-PESSOA", "B-VÍTIMA": "B-PESSOA", "I-SOCIAL": "I-PESSOA",

    # 🚗 Veículo
    "B-VEÍCULO,": "B-VEÍCULO", "I-VEÍCULO,": "I-VEÍCULO", "I-VEÍCULO.": "I-VEÍCULO",
    "I-BENDEREÇO": "I-ENDEREÇO", "C-Veículo": "B-VEÍCULO", "B-CARRO": "B-VEÍCULO",
    "I-CARRO": "I-VEÍCULO", "I-MODELO": "I-VEÍCULO", "I-MARCA": "I-VEÍCULO",

    # 📄 Documentos
    "B-BOLETIM": "O", "B-CFP": "O", "B-RENAVAN": "O", "B-DOCUMENTO": "O",
    "I-DOCUMENTO": "O", "B-CHASSIS": "O", "B-IMEI": "O",

    # 🏢 Organização (ajustes)
    "B-DEPOL": "O", "I-DEPOL": "O", "B-DELEGACIA": "O", "I-DELEGACIA": "O",
    "B-CONTA": "O", "B-AGÊNCIA": "O", "I-AGÊNCIA": "O",

    # 📬 Endereço
    "B-RIO": "B-ENDEREÇO", "I-RIO": "I-ENDEREÇO",

    # 📞 Contato
    "I-CELULAR": "I-TELEFONE",

    # 🧠 Outros
    "B-RELATORIA": "O", "I-RELATORIA": "O", "B-EQUIPE": "O", "I-EQUIPE": "O",
    "B-INTERLOCUTOR": "O", "I-INTERLOCUTOR": "O", "B-NIOP": "O", "I-NIOP": "O",
    "I-KIT": "O", "B-FOLHA": "O", "B-VALOR": "O", "I-VALOR": "O",
    "B-OCORRENCIA": "O", "I-OCORRENCIA": "O", "I-SENSIVEL": "O",
    "B-RELACIONAMENTO": "O", "I-RELACIONAMENTO": "O", "B-FACA": "O", "I-FACA": "O",
    "B-BIBLIA": "O", "B-BOLSA": "O", "I-BOLSA": "O", "B-FAMILIA": "O", "I-FAMILIA": "O",
    "B-GUARDA": "O", "I-GUARDA": "O", "B-GUARNICAO": "O", "I-GUARNICAO": "O",
    "B-BEBE": "O", "I-BEBE": "O", "B-MOTORISTA": "O", "I-MOTORISTA": "O",
    "B-MARIDO": "O", "I-MARIDO": "O", "B-MAE": "O", "I-MAE": "O",
    "O-CASAL": "O", "O...):": "O", "O/:": "O", "O°": "O", "O?": "O", "O);": "O",
    "O-LOCAL": "O",
    "O...)": "O",
    "O/": "O",
    'I-POLICIA': "O",
    'I-POLÍCIA': "O",
    "I-PIX":"I-BANCO",
    "I-PESSOA\"": "I-PESSOA",
    "I-PESSOA”": "I-PESSOA",
    "I-OUTRO": "O",
    "I-ENDEREÇO)":"I-ENDEREÇO", 
    "I-ENDEREÇO.":"I-ENDEREÇO", 
    "I-ESTADO":"O", 
    "I-FOGO":"O", 
    "I-HOSPITAL":"I-EMPRESA", 
    "I-IDADE":"O", 
    "I-MENOR":"O", 
    "I-EMPRESA,":"I-EMPRESA",
    "I-EMPRESA”":"I-EMPRESA",
    "I-CONVERSA": "O",
    "I-BANCO,":"I-BANCO",
    "I-CARTÃO":"I-BANCO",
    "I-CENA":"O",
    "I-CENTIMETRO":"O",
    "I-AMEAÇA":"O",
    "B-TEMPO":"O",
    "B-POLICIA": "O",
    "B-POLÍCIA": "O",
    "B-PIX": "B-BANCO",
    "B-FOGO": "O",
    "B-IDADE": "O",
    "B-MENOR": "O",
    "B-OUTRA": "O",
    "B-OUTRO": "O",
    "B-OUTUBRO": "O",
    "B-EMPRESA,":"B-EMPRESA",
    "B-CONVERSA": "O",
    "B-CENA": "O",
    "B-CENTIMETRO": "O",
    "B-AMEAÇA":"O",
    "B-DEPOENTE":"O",
    "I-DEPOENTE":"O"
  
}

df[tag_column] = df[tag_column].replace(mapeamento_tags_extra)


# Salvar o arquivo atualizado
output_file_path = os.path.join(os.path.dirname(__file__),'..','data','data-chatgpt','IOB_relatos_consolidado_chatgpt_clear.csv')
df.to_csv(output_file_path, index=False)

print(f"\nArquivo atualizado com as tags agrupadas e substituídas, mantendo os prefixos B- e I-.")
print(f"Salvo em: {output_file_path}")

# Exibir estatísticas das tags após o mapeamento
# Exibir estatísticas das tags após o mapeamento (incluindo "O")
tags_apos_mapeamento = df[tag_column].dropna().astype(str).tolist()
tag_counts_apos = Counter(tags_apos_mapeamento)

print("\nDistribuição de tags após o mapeamento (incluindo 'O'):")
for tag, count in sorted(tag_counts_apos.items()):
    print(f"{tag}: {count}")

# Agrupar tokens por entidade (incluindo "O" como categoria própria)
entidades = []
for tag in df[tag_column].dropna().astype(str):
    if tag == "O":
        entidades.append("O")
    elif "-" in tag:
        entidades.append(tag.split("-")[1])

contagem_por_entidade = Counter(entidades)

print("\nQuantidade de tokens por entidade (com 'O'):")
for entidade, count in contagem_por_entidade.items():
    print(f"{entidade}: {count}")
