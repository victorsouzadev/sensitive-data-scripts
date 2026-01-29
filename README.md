# sensitive-data

## 📌 Visão Geral

Este repositório contém o pipeline completo para **preparo, processamento, anotação, padronização e treinamento** de modelos de **Reconhecimento de Entidades Nomeadas (NER)** voltados à **identificação e desidentificação de dados sensíveis**.

O fluxo foi projetado para garantir:

- Qualidade e consistência dos dados  
- Rastreabilidade das etapas  
- Controle de volume  
- Padronização das anotações no formato **IOB**  
- Reprodutibilidade científica  

---

## 🗂️ Estrutura Geral do Pipeline

1. Recorte do dataset original  
2. Seleção de colunas relevantes  
3. Split do dataset em chunks  
4. Extração de entidades sensíveis (NER via LLM)  
5. Consolidação dos resultados
6. Padronização e análise de tags IOB  
7. Converte do formato IOB para JSON
8. Treinamento e avaliação de modelos NER  

---

# 🔹 Preparação e Anotação do Dataset

## ✂️ Recorte de Dados

### Objetivo
Reduzir o dataset original, mantendo apenas registros relevantes ao contexto do estudo.

### Fonte dos Dados
`merged_all_columns_2019_ate_2023-003.csv`

### Script
`dataset/NER/recorte_base_de_dados.ipynb`

### Descrição
- Filtragem por município (ex.: Marabá)  
- Agrupamento por tipo de ocorrência (`consolidado`)  
- Seleção das **10 ocorrências mais frequentes**  
- Preservação de todos os registros associados  

### Saída
`top_10_consolidado_maraba_full.csv`

---

## 🧩 Seleção de Colunas

### Objetivo
Manter apenas campos essenciais para processamento textual e rastreabilidade.

### Script
`dataset/NER/selecao_dados.ipynb`

### Descrição
- Seleção de identificadores e texto do relato  
- Redução de ruído e custo computacional  

### Saída
`top_10_consolidado_maraba_reduzido_colunas.csv`

---

## 🔀 Split do Dataset

### Objetivo
Dividir o dataset em partes menores para evitar problemas de memória e permitir reprocessamento incremental.

### Script
`dataset/NER/split.py`

### Descrição
- Leitura do arquivo em blocos de 1000 registros  
- Geração de arquivos independentes  

### Saída
`split/relatos_chunk_*.csv`

---

## 🧠 Extração de Entidades Sensíveis (NER)

### Objetivo
Identificar automaticamente dados sensíveis utilizando o padrão **IOB**.

### Script
`dataset/NER/generate_openia.py`

### Entidades Identificadas
BANCO, CNH, CPF, EMPRESA, ENDEREÇO, PESSOA, RG, TELEFONE, VEÍCULO, CNPJ, EMAIL

### Descrição
- Envio dos relatos para modelo via API  
- Retorno token a token com rótulos IOB  
- Filtragem apenas de entidades sensíveis  

### Saída
`api_openia_relatos_chunk_*.csv`

---

## 🔗 Consolidação dos Resultados

### Objetivo
Unificar todas as anotações em um único dataset.

### Script
`dataset/NER/unir.py`

### Saída
`IOB_relatos_consolidado_chatgpt.csv`

---

## 🧹 Padronização e Análise de Tags IOB

### Objetivo
Corrigir inconsistências e gerar estatísticas confiáveis das entidades.

### Script
`dataset/NER/get_uniques.py`

### Descrição
- Normalização de tags  
- Remoção de ruídos  
- Preservação do padrão B-/I-  
- Geração de estatísticas  

### Saída
`IOB_relatos_consolidado_chatgpt_clear.csv`

---

## 🧹 Converte IOB para JSON

### Objetivo
Realiza conversão do formato IOB para JSON para ser utilizados no modelos transformers

### Script
`dataset/NER/iob_to_json.py`

### Descrição
- Transforma o formato IOB para JSON

### Saída
`output_chatgpt.json`

---

## ✅ Dataset Final

O dataset final é:

- Consolidado e padronizado  
- Anotado em formato IOB  
- Livre de inconsistências  
- Pronto para:
  - Treinamento de modelos NER  
  - Avaliações comparativas  
  - Desidentificação e anonimização  
  - Pesquisas acadêmicas  

---

# 🔹 Treinamento de Modelos NER

## 🎯 Objetivos

- Identificar dados sensíveis automaticamente  
- Comparar **BiLSTM** e **Transformers (BERTimbau)**  
- Garantir reprodutibilidade e organização por execução  

---

## 📂 Dataset de Entrada

**Formato:** JSON com offsets de caracteres

```json
{
  "doc_id": "ex-001",
  "doc_text": "CPF de João da Silva é 123.456.789-00",
  "entities": [
    { "start_offset": 0, "end_offset": 3, "label": "CPF" }
  ]
}
```

**Arquivo:**  
`dataset/output_chatgpt.json`

---

# 🧠 Treinamento com Transformers (BERTimbau)

## Modelo Base
- `neuralmind/bert-base-portuguese-cased`
- Cabeça de Token Classification

## Split do Dataset
- Treino: 80%  
- Validação: 10%  
- Teste: 10%  

## Configuração

| Parâmetro | Valor |
|---------|------|
| MAX_LEN | 512 |
| EPOCHS | 1 |
| BATCH | 16 |
| Learning Rate | 3e-5 |
| Seed Base | 42 |

## Execuções

Cada execução gera um diretório próprio:

```
runs/
└── <model>__<dataset>__<timestamp>/
    ├── checkpoints/
    ├── eval/
    ├── model/
    └── run_manifest.json
```

---

# 🔁 Treinamento com BiLSTM

## Visão Geral
Modelo clássico baseado em redes recorrentes, com menor custo computacional.

**Script:**  
`dataset/training/ner-using-bidirectional-lstm.ipynb`

---

## 🔬 Comparação entre Modelos

| Aspecto | BiLSTM | Transformer |
|------|------|-----------|
| Pré-treinamento | Não | Sim |
| Custo computacional | Baixo | Alto |
| Contexto | Médio | Alto |
| Desempenho | Bom | Muito alto |
| Hardware | CPU/GPU leve | GPU recomendada |

---

## 💾 Artefatos Gerados

### Transformer
- Modelo treinado  
- Tokenizer  
- Checkpoints  
- Métricas e matrizes de confusão  

### BiLSTM
- Pesos do modelo  
- Histórico de treino  
- Métricas  
- Gráficos  

---

## ⚖️ Benchmark

Comparação direta entre os modelos utilizando textos limpos para desidentificação.

**Script:**  
`dataset/training/ner-using-bidirectional-lstm.ipynb`

---

## 📦 Extração de Artefatos

Geração consolidada de resultados e métricas.

**Script:**  
`dataset/training/results_ner.ipynb`
