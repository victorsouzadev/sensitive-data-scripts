import openai
import re
import csv
import os
from dotenv import load_dotenv
from openai import OpenAI
# ✅ Carregar variáveis do arquivo .env
load_dotenv()

# ✅ Obter a chave da API do ambiente

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

# client = openai.OpenAI(
#     base_url="https://api.groq.com/openai/v1",
#     api_key=os.environ.get("GROQ_API_KEY")
# )


# client = OpenAI(api_key=os.environ.get("PERPLEXITY_API_KEY"), base_url="https://api.perplexity.ai")
# Função para identificar entidades no formato IOB
def identificar_entidades_iob(texto):
    prompt = f"""
     Extraia entidades sensíveis do seguinte texto e retorne no formato IOB (Inside-Outside-Beginning).

    Analise o texto e identifique entidades que contenham informações sensíveis conforme as categorias abaixo. Extraia cada entidade mencionada no contexto, garantindo a preservação do significado original e marcando corretamente sua classificação.

    deve trazer utilizar somente as entidades: BANCO,CNH,CPF,EMPRESA,ENDEREÇO,PESSOA,RG,TELEFONE,VEÍCULO,CNPJ,EMAIL somente identificar essas entidades
    **Texto:** {texto}

    Categorias de Entidades Sensíveis:
    BANCO (Informações Financeiras)
    Identifique números de agência, conta bancária, código do banco, número de cartão, chave Pix e referências diretas a informações bancárias.
    não indetificar nomes de bancos

    Formato esperado:
    Código do banco → Ex: "Banco 341" (Itaú), "Banco 104" (Caixa)
    Agência bancária → Ex: "Agência 1234", "Agência 00325-9"
    Conta bancária → Ex: "Conta 987654-0", "C/C 543210"
    Chave Pix (CPF, CNPJ, telefone, e-mail, aleatória) → Ex: "Pix CPF: 123.456.789-00", "Chave aleatória: a1b2c3d4e5f6"
    Número de cartão de crédito/débito (16 dígitos) → Ex: "Cartão 1234 5678 9012 3456"
    CNH (Carteira Nacional de Habilitação)
    Exemplo: "CNH nº 12345678900", "Minha habilitação é 98765432100".
    CPF (Cadastro de Pessoa Física)
    Exemplo: "CPF 123.456.789-00", "Documento: 98765432100".
    EMPRESA (Nome de Empresas e Organizações)
    Exemplo: "Empresa XPTO Ltda.", "Trabalho na Petrobrás".
    ENDEREÇO (Logradouro, número, cidade, estado, CEP)
    Exemplo: "Rua das Flores, 123", "Av. Paulista, São Paulo - SP", "CEP 01001-000", "Folha (fl) 12, Quadra (qd) 11, Lote (lt) 02".
    PESSOA (Nome de Pessoas e Apelidos)
    Exemplo: "João Silva", "Maria dos Santos".
    RG (Registro Geral de Identidade, identidade, Documento etc.)
    Exemplo: "RG 12.345.678-9", "Meu documento de identidade".
    TELEFONE (Números de telefone, fixo ou celular)
    Exemplo: "(11) 98765-4321", "Telefone: 21-3333-2222".
    VEÍCULO (placas, chassi, renavam)
    Exemplo: "Placa ABC-1234".
    CNPJ (Cadastro Nacional de Pessoa Jurídica)
    Exemplo: "CNPJ 12.345.678/0001-99".
    EMAIL (Endereços de e-mail completos)
    Exemplo: "email@email.com", "contato@empresa.com.br".
    
    **Nome das Entidades**
    Não usar o nome da entidade como entidade e não definir modelos e marcas como entidade, sempre busque por informações sensiveis
    
    **Para veiculos **
    Buscar por informações sensiveis referentes a veículos como placa renavam chassi etc. não usar nome de veiculos ou modelo
    
    **Formato esperado (IOB):**
    Cada palavra do texto deve ser marcada com a entidade correspondente ou "O" se não for entidade.

    **Exemplo de saída:**
    João B-PESSOA
    Silva I-PESSOA
    transferiu O
    dinheiro O
    para O
    o O
    Banco O
    341 B-BANCO
    , I-BANCO
    agência I-BANCO
    1234 I-BANCO
    , I-BANCO
    conta I-BANCO
    567890 I-BANCO
    . O
    CPF O
    123.456.789-00 B-CPF
    . O

    **Retorne apenas o texto no formato IOB, sem explicações adicionais.**
    """

    try:
        resposta = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Você é um assistente que extrai entidades no formato IOB."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        # Verifica se a resposta está vazia
        if not resposta or not resposta.choices:
            print("Erro: Resposta vazia da API.")
            return ""

        resultado = resposta.choices[0].message.content.strip()

        print("🔍 Resposta da API (bruta):\n", resultado)  # Para depuração

        # ✅ Removendo blocos Markdown se existirem
        resultado = re.sub(r"^```[\w]*\n|\n```$", "", resultado.strip())

        return resultado

    except Exception as e:
        print("Erro na API:", e)
        return ""

# Função para salvar resultados em CSV
def salvar_csv(dados, nome_arquivo="entidades_relatos.csv"):
    with open(nome_arquivo, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "Palavra", "Tag IOB"])  # Cabeçalhos

        for row_id, resultado in dados.items():
            for linha in resultado.split("\n"):
                if linha.strip():
                    partes = linha.split()
                    if len(partes) == 2:
                        palavra, tag = partes
                        writer.writerow([row_id, palavra, tag])

    print(f"📁 Arquivo salvo: {nome_arquivo}")

# Função para processar os relatos do CSV
def processar_relatos(arquivo_csv, limite=1000):
    resultados = {}

    with open(arquivo_csv, mode="r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Ignorar cabeçalho

        for index, row in enumerate(reader):
            if index >= limite:  # Processar apenas os primeiros 100 relatos
                break

            row_id, relato = row
            print(f"\n🔹 Processando relato {row_id} ({index+1}/{limite})...")

            # Extrair entidades no formato IOB
            entidades_iob = identificar_entidades_iob(relato)
            resultados[row_id] = entidades_iob

    return resultados


# Teste do script
if __name__ == "__main__":
        # Defina a pasta onde estão os arquivos divididos (split)
    pasta_entrada = os.path.join(os.path.dirname(__file__), "split")

    # Listar todos os arquivos CSV dentro da pasta
    arquivos_entrada = [os.path.join(pasta_entrada, f) for f in os.listdir(pasta_entrada) if f.endswith(".csv")]

    print(arquivos_entrada)
    print("\n📌 Iniciando processamento de múltiplos arquivos...")

    file_name = ['relatos_chunk_1.csv']
    for file in file_name:
        arquivo_entrada = os.path.join(os.path.dirname(__file__),'split', file)
        print(arquivo_entrada)
    
        # print("\n📌 Iniciando processamento dos relatos...")
        entidades_relatos = processar_relatos(arquivo_entrada)

        if entidades_relatos:
            salvar_csv(entidades_relatos,"api_openia_"+file)
