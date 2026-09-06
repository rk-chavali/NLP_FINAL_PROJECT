# Multi-Label Petition Classification (EUROVOC)

Classifies petition text into EUROVOC subject categories, a multi-label problem
where any document can carry several categories at once. Four models are
compared, from a TF-IDF baseline to fine-tuned BERT variants, and served through
a Streamlit app.

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗_Transformers-FFD21E)](https://huggingface.co/docs/transformers)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-22C55E.svg)](LICENSE)

## Results on EURLEX

| Model | F1 | Precision | Recall |
|-------|-----|-----------|--------|
| Naive Bayes | 0.290 | 0.833 | 0.176 |
| **Passive Aggressive** | **0.688** | 0.765 | 0.625 |
| BERT + GRU | 0.529 | 0.860 | 0.382 |
| BERT + BiLSTM | 0.298 | 0.855 | 0.181 |

**The TF-IDF Passive Aggressive classifier beats both BERT variants.** That is
the interesting result, and it is worth reading the reason off the columns rather
than the headline number.

Every model has high precision and poor recall. They are confident about the
labels they assign and assign too few of them, which is the classic failure mode
for multi-label problems with a long tail of rare categories: predicting the
common labels and abstaining on the rest is locally rewarded. Passive Aggressive
wins because its recall is 0.625 while the next best manages 0.382, not because
it is more precise. It is in fact the least precise of the four.

The BERT models were trained on limited data and epochs. With more of both, the
ordering would likely change. As it stands, a linear model on TF-IDF features is
the better choice here, which is a useful reminder that transformer capacity
needs the data budget to match.

## Models

| Approach | Detail |
|----------|--------|
| Naive Bayes | Multinomial NB over TF-IDF, one-vs-rest |
| Passive Aggressive | Online linear classifier over TF-IDF, one-vs-rest |
| BERT + GRU | BERT embeddings into a GRU head |
| BERT + BiLSTM | BERT embeddings into a bidirectional LSTM head |

Labels are binarised with `MultiLabelBinarizer` so each category becomes an
independent binary decision.

## Run it

```bash
git clone https://github.com/rk-chavali/NLP_FINAL_PROJECT
cd NLP_FINAL_PROJECT
pip install -r requirements.txt
streamlit run streamlit-app.py
```

Opens on http://localhost:8501. Enter petition text or load a sample, pick a
model in the sidebar, and classify.

## Model weights are not included

GitHub's file size limits keep the `.pkl` and `.pt` weights out of the repo, so
the app will not run until you export them from `codebase.ipynb`:

```python
joblib.dump(tfidf, "tfidf_vectorizer.pkl")
joblib.dump(mlb, "multilabel_binarizer.pkl")
joblib.dump(model_nb, "naive_bayes_model.pkl")
joblib.dump(model_pa, "passive_aggressive_model.pkl")
torch.save(model_bert_gru, "bert_gru_model.pt")
torch.save(model_bilstm, "bert_bilstm_model.pt")
```

Place all six files next to `streamlit-app.py`.

## Layout

| Path | Purpose |
|------|---------|
| `codebase.ipynb` | Training, evaluation, and weight export |
| `streamlit-app.py` | Inference UI |
| `requirements.txt` | Dependencies |

## License

MIT, see [LICENSE](LICENSE).
