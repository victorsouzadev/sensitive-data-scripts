FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Sistema + Python
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip git curl \
 && ln -sf python3.10 /usr/bin/python3 \
 && ln -sf pip3 /usr/bin/pip \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Se você usa requirements.txt, copie primeiro
COPY requirements.txt .

# Pip e libs principais em ordem (sem '!' e com pins)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir "numpy<2.0" \
 # PyTorch compatível com CUDA 12.1
 && pip install --no-cache-dir torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
 # O que você pediu para "incluir"
 && pip install --no-cache-dir scikit-learn datasets==3.6.0 "transformers>=4.41" accelerate peft \
 && pip install --no-cache-dir "fsspec==2025.3.0" "gcsfs==2025.3.0" \
 # Pacotes extras do seu requirements (se houver)
 && pip install --no-cache-dir -r requirements.txt \
 # Opcional: limpar cache do pip para reduzir a imagem
 && pip cache purge

# Código
COPY . .

CMD ["python3", "main.py"]
