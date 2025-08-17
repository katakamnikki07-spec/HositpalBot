# HositpalBot

# Medical QA Bot — Transformer + RAG

## Overview
This project implements a medical question-answering system trained on the provided dataset (`mle_screening_dataset.csv`).  
It uses a **Seq2Seq Transformer model with SentencePiece tokenization** combined with a **retriever (Sentence-Transformers + FAISS, with TF-IDF fallback)** to answer medical questions.


---

## Approach

1. **Data Preprocessing**
   - Loaded dataset with `question` and `answer` columns.
   - Normalized text and created train/validation/test splits.
   - Built custom SentencePiece tokenizer (BPE, vocab=8000).

2. **Model Training**
   - Implemented a **Seq2Seq Transformer** with positional embeddings.
   - Trained on Q–A pairs for up to 60 epochs with early stopping.
   - Used **CrossEntropyLoss** with teacher forcing.

3. **Retriever**
   - Dense retrieval using `sentence-transformers/all-MiniLM-L6-v2` + FAISS.
   - Fallback to TF-IDF + cosine similarity if embeddings unavailable.
   - Supports **RAG mode**: retrieved passages are prepended to the source question.

4. **Evaluation**
   - **ROUGE-L** and **BLEU** for generative answers.
   - Example scores (subset of test set):
     - Generator only → ROUGE-L ~0.32, BLEU ~18
     - RAG (k=3) → ROUGE-L ~0.41, BLEU ~24

5. **Interactive Demo**
   - Built with **Gradio**.
   - Two modes: `Generator only` and `RAG (retrieve + generate)`.
   - Option to display retrieved sources.

---

## Strengths
- Combines **generative QA** with **retrieval-augmented generation (RAG)**.
- Supports **interactive chatbot** for medical QAs.
- Evaluated with multiple metrics.

## Limitations
- Small dataset limits generalization.
- Seq2Seq transformer is lightweight — may underperform vs large LLMs.
- Retrieval depends on dataset coverage.

## Potential Improvements
- Fine-tune **BioBERT** or **T5-small** for improved QA.
- Use **Dense Passage Retrieval (DPR)** instead of FAISS flat index.
- Extend dataset with public medical QA sources (e.g., MedQuAD).

---

## Files
- `train_medical_qa.ipynb` — Notebook with training, evaluation, and demo.
- `medical_qa.pt` — Model checkpoint (saved after training).
- `spm_medical.model` — SentencePiece tokenizer.
- `README.md` — This file.

---

## Submission Instructions
Upload to GitHub with:
1. **Notebook/script** (`train_medical_qa.ipynb` or `.py`)
2. **README.md**
3. **Saved checkpoint**
