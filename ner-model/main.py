# ------------ 1. IMPORTS & CONFIG -----------------
import os, re, json, numpy as np, torch, matplotlib.pyplot as plt
from datetime import datetime
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForTokenClassification,
    DataCollatorForTokenClassification, TrainingArguments, Trainer, pipeline,
    AutoModelForCausalLM
)
from datasets import Dataset, DatasetDict
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import random
print(torch.cuda.mem_get_info()) 
# 💾 Caminhos
LOCAL_PATH = ''
DATASET_PRIMEIRO_HAREM = os.path.join(LOCAL_PATH, 'dataset', 'output_chatgpt.json')

# 🎯 MODELO
MODEL_ID = "neuralmind/bert-base-portuguese-cased"

# Hiperparâmetros
MAX_LEN = 512
EPOCHS = 1
BATCH = 16
NUM_RUNS = 1   # <<<<<<<<<<<<<<<< número de iterações

# === util: seeds para reprodutibilidade parcial ===
BASE_SEED = 42

# === NAMING DINÂMICO (modelo + dataset + timestamp) ===
def slugify(text: str) -> str:
    text = text.split("/")[-1]  # última parte do repo (ex.: Tucano-160m)
    text = text.replace("_", "-").replace(" ", "-")
    return re.sub(r"[^A-Za-z0-9\-]+", "-", text).strip("-").lower()

MODEL_SLUG = slugify(MODEL_ID)
DATASET_NAME = os.path.splitext(os.path.basename(DATASET_PRIMEIRO_HAREM))[0]  # ex.: output_chatgpt
BASE_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")  # base comum a todas as iterações

# ------------ 2. UTILIDADES -----------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_dataset(data_path):
    """Carrega o JSON anotado em nível de caractere ⇒ lista de dicts."""
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    ds = []
    for doc in data:
        text = doc["doc_text"]
        char_labels = ["O"] * len(text)
        for ent in doc["entities"]:
            start, end, lbl = ent["start_offset"], ent["end_offset"], ent["label"]
            char_labels[start] = f"B-{lbl}"
            for i in range(start + 1, end):
                char_labels[i] = f"I-{lbl}"
        ds.append({"id": doc.get("doc_id", ""), "text": text, "char_labels": char_labels})
    return ds

def get_label_list(dataset):
    return sorted({lbl for ex in dataset for lbl in ex["char_labels"]})

def tokenize_and_align_labels(examples, tokenizer, label2id):
    """Converte lista[dict] → dict de listas para o HuggingFace Dataset."""
    input_ids, attn_masks, label_ids = [], [], []
    for ex in examples:
        enc = tokenizer(
            ex["text"], padding="max_length", truncation=True,
            max_length=MAX_LEN, return_offsets_mapping=True
        )
        labels = []
        for offs in enc["offset_mapping"]:
            if offs[0] == offs[1]:
                labels.append(-100)  # tokens especiais
            else:
                char_pos = min(offs[0], len(ex["char_labels"]) - 1)
                labels.append(label2id.get(ex["char_labels"][char_pos], label2id["O"]))
        input_ids.append(enc["input_ids"])
        attn_masks.append(enc["attention_mask"])
        label_ids.append(labels)
    return {"input_ids": input_ids, "attention_mask": attn_masks, "labels": label_ids}

def ensure_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)

# ------------ 3. CARREGAR DADOS & LABELS (uma única vez) ------------
raw_dataset = load_dataset(DATASET_PRIMEIRO_HAREM)
label_list   = get_label_list(raw_dataset)
label2id     = {l: i for i, l in enumerate(label_list)}
id2label     = {i: l for l, i in label2id.items()}

# split fixo (usa a mesma divisão para todas as iterações)
train_sz = int(0.8 * len(raw_dataset))
val_sz   = int(0.1 * len(raw_dataset))
test_sz  = len(raw_dataset) - train_sz - val_sz
train_ds = raw_dataset[:train_sz]
val_ds   = raw_dataset[train_sz:train_sz+val_sz]
test_ds  = raw_dataset[train_sz+val_sz:]

# ------------ 4. LOOP DE 5 EXECUÇÕES -------------------
for run_idx in range(1, NUM_RUNS + 1):
    print("\n" + "="*80)
    print(f"🚀 Iniciando iteração {run_idx}/{NUM_RUNS}")
    RUN_STAMP = f"{BASE_STAMP}-{run_idx:02d}"
    RUN_NAME  = f"{MODEL_SLUG}__{DATASET_NAME}__{RUN_STAMP}"

    # 🌿 Estrutura de saídas por execução
    BASE_OUT = os.path.join("/raid/aluno_luan/ner-project", "runs", RUN_NAME)
    OUTPUT_DIR = os.path.join(BASE_OUT, "checkpoints")   # checkpoints do Trainer
    EVAL_DIR   = os.path.join(BASE_OUT, "eval")          # avaliações/relatórios
    MODEL_SAVE_PATH = os.path.join(BASE_OUT, "model")    # artefatos finais do modelo
    ensure_dirs(OUTPUT_DIR, EVAL_DIR, MODEL_SAVE_PATH)

    print(f"🔧 RUN_NAME: {RUN_NAME}")
    print(f"   BASE_OUT: {BASE_OUT}")
    print(f"   OUTPUT_DIR (checkpoints): {OUTPUT_DIR}")
    print(f"   EVAL_DIR: {EVAL_DIR}")
    print(f"   MODEL_SAVE_PATH: {MODEL_SAVE_PATH}")

    # Seed diferente por iteração (mas determinístico)
    seed_this = BASE_SEED + run_idx
    set_seed(seed_this)

    # ------------ 5. TOKENIZER & DATASETS (tokeniza a cada run) -----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})

    train_tok = tokenize_and_align_labels(train_ds, tokenizer, label2id)
    val_tok   = tokenize_and_align_labels(val_ds,   tokenizer, label2id)
    test_tok  = tokenize_and_align_labels(test_ds,  tokenizer, label2id)

    dataset = DatasetDict({
        "train": Dataset.from_dict(train_tok),
        "validation": Dataset.from_dict(val_tok),
        "test": Dataset.from_dict(test_tok)
    })

    # ------------ 6. MODELO Tucano + Cabeça NER -----------
    config = AutoConfig.from_pretrained(
        MODEL_ID,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        pad_token_id=tokenizer.pad_token_id
    )

    model = AutoModelForTokenClassification.from_pretrained(   # ⬅️ trocado!
        MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,
        use_safetensors=True
    )
    model.resize_token_embeddings(len(tokenizer))


    # ------------ 7. TRAINING ARGUMENTS & TRAINER ---------
    data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,      # pasta da iteração
        do_train=True,
        do_eval=True,
        logging_steps=5000,
        save_steps=5000,
        eval_steps=5000,
        learning_rate=3e-5,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH,
        per_device_eval_batch_size=BATCH,
        fp16=torch.cuda.is_available(),
        weight_decay=0.01,
        save_total_limit=2,
        report_to="none",
        seed=seed_this
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # ------------ 8. TREINO -------------------------------
    trainer.train()

    # ------------ 9. AVALIAÇÃO (rápida) -------------------
    pred_logits, true_ids, _ = trainer.predict(dataset["test"])
    pred_ids = np.argmax(pred_logits, axis=2)

    # reconstruir seq. rotuladas retirando -100
    true_bio, pred_bio = [], []
    for t_row, p_row in zip(true_ids, pred_ids):
        for t, p in zip(t_row, p_row):
            if t != -100:
                true_bio.append(id2label[t])
                pred_bio.append(id2label[p])

    def strip_prefix(tags):
        return [t.replace("B-","").replace("I-","") for t in tags]

    true_plain = strip_prefix(true_bio)
    pred_plain = strip_prefix(pred_bio)

    print("=== RELATÓRIO COM B-/I- ===")
    print(classification_report(true_bio, pred_bio, labels=label_list))

    print("\n=== RELATÓRIO SEM B-/I- ===")
    plain_labels = sorted(list(set(true_plain + pred_plain)))
    print(classification_report(true_plain, pred_plain, labels=plain_labels))

    # Matriz confusão (sem prefixos)
    cm_quick = confusion_matrix(true_plain, pred_plain, labels=plain_labels)
    plt.figure(figsize=(10,8))
    ConfusionMatrixDisplay(cm_quick, display_labels=plain_labels).plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"Matriz de Confusão – {MODEL_SLUG} (sem B-/I-)")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"confusion_matrix__{MODEL_SLUG}__plain.png"))
    plt.close()

    # ------------ 9b. AVALIAÇÃO (detalhada) ---------------
    predictions, true_label_ids, _ = trainer.predict(dataset["test"])
    predicted_label_ids = np.argmax(predictions, axis=2)

    true_labels_bio, pred_labels_bio = [], []
    for t_ids, p_ids in zip(true_label_ids, predicted_label_ids):
        t_seq, p_seq = [], []
        for t_id, p_id in zip(t_ids, p_ids):
            if t_id != -100:
                t_seq.append(id2label.get(t_id, "O"))
                p_seq.append(id2label.get(p_id, "O"))
        m = min(len(t_seq), len(p_seq))
        true_labels_bio.append(t_seq[:m])
        pred_labels_bio.append(p_seq[:m])

    def strip_bio_prefix(tags_nested):
        return [[tag.replace("B-", "").replace("I-", "") for tag in sent] for sent in tags_nested]

    true_labels_plain = strip_bio_prefix(true_labels_bio)
    pred_labels_plain = strip_bio_prefix(pred_labels_bio)

    # Flatten ignorando "O"
    flat_true_bio, flat_pred_bio = [], []
    for t_seq, p_seq in zip(true_labels_bio, pred_labels_bio):
        for t, p in zip(t_seq, p_seq):
            if t != "O":
                flat_true_bio.append(t)
                flat_pred_bio.append(p)

    flat_true_plain, flat_pred_plain = [], []
    for t_seq, p_seq in zip(true_labels_plain, pred_labels_plain):
        for t, p in zip(t_seq, p_seq):
            if t != "O":
                flat_true_plain.append(t)
                flat_pred_plain.append(p)

    labels_bio   = sorted(list(set(flat_true_bio   + flat_pred_bio)))
    labels_plain = sorted(list(set(flat_true_plain + flat_pred_plain)))

    report_bio   = classification_report(flat_true_bio,   flat_pred_bio,   labels=labels_bio)
    report_plain = classification_report(flat_true_plain, flat_pred_plain, labels=labels_plain)

    print("\n=== RELATÓRIO COM PREFIXOS B-/I- (filtrado) ===")
    print(report_bio)
    print("\n=== RELATÓRIO SEM PREFIXOS B-/I- (filtrado) ===")
    print(report_plain)

    # Matriz de Confusão (COM prefixos)
    cm_bio = confusion_matrix(flat_true_bio, flat_pred_bio, labels=labels_bio)
    fig, ax = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(cm_bio, display_labels=labels_bio).plot(cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title(f"Matriz de Confusão - {MODEL_SLUG} (com B-/I-)")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"confusion_matrix__{MODEL_SLUG}__bio.png"))
    plt.close(fig)

    # Matriz de Confusão (SEM prefixos)
    cm_plain = confusion_matrix(flat_true_plain, flat_pred_plain, labels=labels_plain)
    fig, ax = plt.subplots(figsize=(10, 8))
    ConfusionMatrixDisplay(cm_plain, display_labels=labels_plain).plot(cmap="Blues", ax=ax, xticks_rotation=45)
    plt.title(f"Matriz de Confusão - {MODEL_SLUG} (sem B-/I-)")
    plt.tight_layout()
    plt.savefig(os.path.join(EVAL_DIR, f"confusion_matrix__{MODEL_SLUG}__plain.png"))
    plt.close(fig)

    # ------------ 10. SALVAR RESULTADOS (nomes padronizados) ----------
    def salvar_json(obj, filename):
        with open(os.path.join(EVAL_DIR, filename), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)

    # 📝 Predições e rótulos
    salvar_json(true_labels_bio,   f"true_labels_bio__{MODEL_SLUG}.json")
    salvar_json(pred_labels_bio,   f"pred_labels_bio__{MODEL_SLUG}.json")
    salvar_json(true_labels_plain, f"true_labels_plain__{MODEL_SLUG}.json")
    salvar_json(pred_labels_plain, f"pred_labels_plain__{MODEL_SLUG}.json")

    # 🧠 Mapeamentos
    salvar_json(label2id, f"label2id__{MODEL_SLUG}.json")
    salvar_json(id2label, f"id2label__{MODEL_SLUG}.json")

    # 📄 Relatórios como texto
    with open(os.path.join(EVAL_DIR, f"classification_report_bio__{MODEL_SLUG}.txt"), "w", encoding="utf-8") as f:
        f.write(report_bio)

    with open(os.path.join(EVAL_DIR, f"classification_report_plain__{MODEL_SLUG}.txt"), "w", encoding="utf-8") as f:
        f.write(report_plain)

    print(f"\n✅ Resultados salvos em: {EVAL_DIR}")

    # ------------ 11. SALVAR MODELO E TOKENIZER -----------
    model.save_pretrained(MODEL_SAVE_PATH)
    tokenizer.save_pretrained(MODEL_SAVE_PATH)

    # 🔖 Manifest com contexto da execução
    manifest = {
        "run_index": run_idx,
        "seed": seed_this,
        "model_id_base": MODEL_ID,
        "model_slug": MODEL_SLUG,
        "dataset_name": DATASET_NAME,
        "run_stamp": RUN_STAMP,
        "run_name": RUN_NAME,
        "paths": {
            "base_out": BASE_OUT,
            "checkpoints": OUTPUT_DIR,
            "eval": EVAL_DIR,
            "model": MODEL_SAVE_PATH
        },
        "hparams": {
            "max_len": MAX_LEN,
            "epochs": EPOCHS,
            "batch": BATCH
        },
        "labels": {
            "label_list": label_list
        }
    }
    with open(os.path.join(BASE_OUT, f"run_manifest__{RUN_NAME}.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"✅ Modelo e tokenizer salvos em: {MODEL_SAVE_PATH}")
    print(f"🗂️ Manifest da execução: {os.path.join(BASE_OUT, f'run_manifest__{RUN_NAME}.json')}")

    # Libera memória GPU entre runs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print("\n🏁 Todas as iterações concluídas.")
