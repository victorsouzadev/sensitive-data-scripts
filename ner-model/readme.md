# NER Trainer – Docker (CUDA 12.1.1)

Treinamento e avaliação de NER a partir de JSON com offsets de caracteres, usando Hugging Face e PyTorch em GPU.  
A imagem é baseada em **`nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04`** e instala Python 3.10 + stack de ML com versões compatíveis.

## ✨ Destaques

- Docker com **CUDA 12.1.1 + cuDNN8**
- **PyTorch 2.2.2 (cu121)**, `transformers>=4.41`, `datasets==3.6.0`
- Pins de **`fsspec==2025.3.0`** e **`gcsfs==2025.3.0`**
- Suporte a **múltiplas iterações** controladas por `NUM_RUNS`
- Saídas organizadas por *run* (checkpoints, relatórios, matrizes de confusão, manifest)
- **Inferência** ao final de cada execução com `pipeline("token-classification")` e `aggregation_strategy="simple"`

## 📦 Requisitos

- **Docker** 24+
- **NVIDIA Driver** e **NVIDIA Container Toolkit**

## 🏗️ Build

```bash
docker build -t ner-trainer .
```

## ▶️ Execução rápida

```bash
docker run --rm   --name aluno_luan   --memory="16g"   --cpus="8.0"   --gpus '"device=1"'   -v /raid/dataset:/workspace/dataset   -v /raid/checkpoints:/workspace/tucano160   -v $(pwd):/workspace   ner-trainer:latest
```

## 🗂️ Estrutura esperada

```
/raid/
├── dataset/
│   └── output_chatgpt.json
└── checkpoints/
repo/
├── Dockerfile
├── requirements.txt
├── main.py
└── README.md
```

## 📥 Formato do dataset (JSON)

```json
[
  {
    "doc_id": "ex-001",
    "doc_text": "CPF de João da Silva é 123.456.789-00...",
    "entities": [
      { "start_offset": 0, "end_offset": 3, "label": "CPF" }
    ]
  }
]
```

## ⚙️ Hiperparâmetros

| Parâmetro | Valor | Descrição |
|------------|--------|------------|
| `MODEL_ID` | `CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it` | Modelo base |
| `MAX_LEN` | 512 | Comprimento máximo |
| `EPOCHS` | 5 | Número de épocas |
| `BATCH` | 2 | Tamanho do batch |
| `NUM_RUNS` | 1 | Iterações de execução |
| `BASE_SEED` | 42 | Semente base |

## 🔁 Saídas

- `checkpoints/` – pontos de treino
- `eval/` – relatórios, matrizes de confusão e métricas
- `model/` – modelo/tokenizer final
- `run_manifest__*.json` – metadados da execução

## 🧠 Inferência

O script realiza inferência automática ao final:

```python
exemplo = "CPF de João da Silva é 123.456.789-00 e o RG 1.234.567-8."
print(ner_pipe(exemplo))
```

## 🛠️ Solução de problemas

- **Driver NVIDIA** ausente → instale `nvidia-container-toolkit`
- **OOM** → reduza `BATCH` ou aumente memória
- **Permissões** → use `--user $(id -u):$(id -g)`
