# SEED_MANIFEST

**Frozen:** 2026-05-12T22:45:58.977352+00:00
**Storage:** `faissprod` / container `v2-seed`
**Encryption:** Microsoft-managed keys (default for Standard_LRS)
**Versioning:** Enabled
**Soft-delete:** 30 days (blob + container)

## Why this exists

Upstream MIMIC-IV access was lost. The 1000-row CSV samples and
everything derived from them (joined link tables, gold chunked
documents, winning FAISS index) are the only surviving copies of
research-time work. This manifest is the data-lineage record
until a managed catalog is in place.

## Summary

- Total files: **139**
- Total size: **6820.6 MB**
- Uploaded this run: 139
- Skipped (already present): 0
- Errors: 0

## Provenance notes

- `gold/chunked_docs.pkl` was produced with embedding model `biomedbert`.
- `prod-index/faiss_winner/` is the FAISS index produced by `biomedbert` —
  selected as the winner from the 54-combo (9 emb x 6 LLM) evaluation grid.
- `dev-indexes/` holds the other 8 FAISS indexes, kept only to reproduce the
  benchmark grid. Production retrieval uses `prod-index/` only.
- `models/` holds HF checkpoints. Re-downloadable from HuggingFace if lost,
  but mirrored for offline / air-gapped runs.

## gold (1 files, 53.1 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `gold/chunked_docs.pkl` | 53.15 | `a24602e5f2812bf5…` |

## silver (3 files, 109.6 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `silver/admissions_df.pkl` | 0.12 | `00e8a25e49b659e0…` |
| `silver/grouped_tables.pkl` | 55.88 | `f114c76e56da6187…` |
| `silver/link_tables.pkl` | 53.65 | `f70e0f2a2e1e2b9e…` |

## bronze (18 files, 75.2 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `bronze/admissions.csv_sample1000.csv` | 0.17 | `7aad430d130dcab3…` |
| `bronze/d_hcpcs.csv.csv` | 3.42 | `4b1ad286079dcbf9…` |
| `bronze/d_icd_diagnoses.csv.csv` | 8.76 | `cb05a1eae6786453…` |
| `bronze/d_icd_procedures.csv.csv` | 7.20 | `32821fa24a03695a…` |
| `bronze/d_labitems.csv.csv` | 0.06 | `4210b5c0744881eb…` |
| `bronze/diagnoses_icd.csv_sample1000.csv` | 0.32 | `da1f678704067c3a…` |
| `bronze/drgcodes.csv_sample1000.csv` | 0.10 | `b411b42508d70ec4…` |
| `bronze/emar.csv_sample1000.csv` | 12.06 | `349635932841f612…` |
| `bronze/hcpcsevents.csv_sample1000.csv` | 0.02 | `c6b40c72881b930e…` |
| `bronze/labevents.csv_sample1000.csv` | 18.59 | `b576eef90b2dc587…` |
| `bronze/microbiologyevents.csv_sample1000.csv` | 0.85 | `06ad481c87fb0b07…` |
| `bronze/pharmacy.csv_sample1000.csv` | 7.29 | `6d14615bf0628f63…` |
| `bronze/poe.csv_sample1000.csv` | 9.19 | `40a1e0cfb98e4820…` |
| `bronze/prescriptions.csv_sample1000.csv` | 6.44 | `e5a137923dc324a7…` |
| `bronze/procedures_icd.csv_sample1000.csv` | 0.06 | `2f8777a20d8bb1c0…` |
| `bronze/provider.csv.csv` | 0.32 | `cef3741b4fa12d86…` |
| `bronze/services.csv_sample1000.csv` | 0.05 | `9fe927c914da95ec…` |
| `bronze/transfers.csv_sample1000.csv` | 0.31 | `5d2aa69a57dea240…` |

## prod-index (2 files, 367.3 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `prod-index/faiss_winner/index.faiss` | 308.70 | `91a37df5bf81fe8c…` |
| `prod-index/faiss_winner/index.pkl` | 58.55 | `75fd32c859ffc5cd…` |

## winner-model (11 files, 418.6 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `models/winner/1_Pooling/config.json` | 0.00 | `ec0eb6432a18121e…` |
| `models/winner/README.md` | 0.00 | `f0b6105c44b7b6d7…` |
| `models/winner/config.json` | 0.00 | `d1eb7203c4ef7beb…` |
| `models/winner/config_sentence_transformers.json` | 0.00 | `35754e70788b133e…` |
| `models/winner/model.safetensors` | 417.66 | `6debbaad3ff02e9d…` |
| `models/winner/modules.json` | 0.00 | `27e3a229cecaf4d6…` |
| `models/winner/sentence_bert_config.json` | 0.00 | `9e8f7a3ce4e41e26…` |
| `models/winner/special_tokens_map.json` | 0.00 | `3c3507f36dff57bc…` |
| `models/winner/tokenizer.json` | 0.67 | `9355eae89d401cee…` |
| `models/winner/tokenizer_config.json` | 0.00 | `f8ce43799207aeb3…` |
| `models/winner/vocab.txt` | 0.22 | `79489a52be45e6fa…` |

## dev-indexes (16 files, 2783.7 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `dev-indexes/faiss_mimic_sample1000_BioBERT/index.faiss` | 308.70 | `e1afb5220b8a8555…` |
| `dev-indexes/faiss_mimic_sample1000_BioBERT/index.pkl` | 58.55 | `ef87e87f1782f954…` |
| `dev-indexes/faiss_mimic_sample1000_BioLORD/index.faiss` | 308.70 | `f5fb15b672c5d2a2…` |
| `dev-indexes/faiss_mimic_sample1000_BioLORD/index.pkl` | 58.55 | `062a6eeaedcd7abd…` |
| `dev-indexes/faiss_mimic_sample1000_MedQuAD/index.faiss` | 308.70 | `415b0fcc3a713584…` |
| `dev-indexes/faiss_mimic_sample1000_MedQuAD/index.pkl` | 58.55 | `ec785fdd17498813…` |
| `dev-indexes/faiss_mimic_sample1000_e5-base/index.faiss` | 308.70 | `1670b3c4109e26e2…` |
| `dev-indexes/faiss_mimic_sample1000_e5-base/index.pkl` | 58.55 | `ab64d194bc32d603…` |
| `dev-indexes/faiss_mimic_sample1000_mini-lm/index.faiss` | 154.35 | `99e25d5300e65299…` |
| `dev-indexes/faiss_mimic_sample1000_mini-lm/index.pkl` | 58.55 | `3b1600e0d1519ffc…` |
| `dev-indexes/faiss_mimic_sample1000_mpnet-v2/index.faiss` | 308.70 | `5c732c3173d24c3c…` |
| `dev-indexes/faiss_mimic_sample1000_mpnet-v2/index.pkl` | 58.55 | `69da29e3142dc087…` |
| `dev-indexes/faiss_mimic_sample1000_ms-marco/index.faiss` | 308.70 | `74ce5be620be6484…` |
| `dev-indexes/faiss_mimic_sample1000_ms-marco/index.pkl` | 58.55 | `21ed183cef90c814…` |
| `dev-indexes/faiss_mimic_sample1000_multi-qa/index.faiss` | 308.70 | `74c7aa4170b85523…` |
| `dev-indexes/faiss_mimic_sample1000_multi-qa/index.pkl` | 58.55 | `f272d3229e5c0ef8…` |

## other-models (88 files, 3013.1 MB)

| blob | size (MB) | sha256 |
|---|---|---|
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/README.md` | 0.00 | `2457a30d25375635…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/config.json` | 0.00 | `4e11e74d17ca8e15…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/model.safetensors` | 413.19 | `7391f4898042f3ac…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/modules.json` | 0.00 | `27e3a229cecaf4d6…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/sentence_bert_config.json` | 0.00 | `5554ea50ca2aef2b…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/special_tokens_map.json` | 0.00 | `5aa43c2f985a2529…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/tokenizer.json` | 0.64 | `95aec636f878d65a…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/tokenizer_config.json` | 0.00 | `4e24025cec6bd44d…` |
| `models/BioBERT-mnli-snli-scinli-scitail-mednli-stsb/vocab.txt` | 0.20 | `eeaa9875b23b04b4…` |
| `models/BioLORD-2023-C/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/BioLORD-2023-C/README.md` | 0.00 | `efaad711384411f7…` |
| `models/BioLORD-2023-C/config.json` | 0.00 | `732db8dfa09066dd…` |
| `models/BioLORD-2023-C/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/BioLORD-2023-C/model.safetensors` | 417.68 | `f6bb11dd5fdae16d…` |
| `models/BioLORD-2023-C/modules.json` | 0.00 | `27e3a229cecaf4d6…` |
| `models/BioLORD-2023-C/sentence_bert_config.json` | 0.00 | `0d36864e76a49566…` |
| `models/BioLORD-2023-C/special_tokens_map.json` | 0.00 | `efa8db3ea1576eeb…` |
| `models/BioLORD-2023-C/tokenizer.json` | 0.68 | `822f9e32e85735fe…` |
| `models/BioLORD-2023-C/tokenizer_config.json` | 0.00 | `6782a4e1dba85f60…` |
| `models/BioLORD-2023-C/vocab.txt` | 0.22 | `dbd90cb94e2247bd…` |
| `models/S-PubMedBert-MS-MARCO/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/S-PubMedBert-MS-MARCO/README.md` | 0.00 | `9ed4c713dceacd45…` |
| `models/S-PubMedBert-MS-MARCO/config.json` | 0.00 | `d1eb7203c4ef7beb…` |
| `models/S-PubMedBert-MS-MARCO/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/S-PubMedBert-MS-MARCO/model.safetensors` | 417.66 | `c2d0567ef02294d4…` |
| `models/S-PubMedBert-MS-MARCO/modules.json` | 0.00 | `27e3a229cecaf4d6…` |
| `models/S-PubMedBert-MS-MARCO/sentence_bert_config.json` | 0.00 | `814645821a8acc2d…` |
| `models/S-PubMedBert-MS-MARCO/special_tokens_map.json` | 0.00 | `5aa43c2f985a2529…` |
| `models/S-PubMedBert-MS-MARCO/tokenizer.json` | 0.67 | `f47527fce7dab0a1…` |
| `models/S-PubMedBert-MS-MARCO/tokenizer_config.json` | 0.00 | `bcadc59593710ac6…` |
| `models/S-PubMedBert-MS-MARCO/vocab.txt` | 0.22 | `79489a52be45e6fa…` |
| `models/S-PubMedBert-MedQuAD/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/S-PubMedBert-MedQuAD/README.md` | 0.00 | `dfe39a1c641f09a0…` |
| `models/S-PubMedBert-MedQuAD/config.json` | 0.00 | `d1eb7203c4ef7beb…` |
| `models/S-PubMedBert-MedQuAD/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/S-PubMedBert-MedQuAD/model.safetensors` | 417.66 | `56a2ca7a2e420fa5…` |
| `models/S-PubMedBert-MedQuAD/modules.json` | 0.00 | `27e3a229cecaf4d6…` |
| `models/S-PubMedBert-MedQuAD/sentence_bert_config.json` | 0.00 | `0d36864e76a49566…` |
| `models/S-PubMedBert-MedQuAD/special_tokens_map.json` | 0.00 | `5aa43c2f985a2529…` |
| `models/S-PubMedBert-MedQuAD/tokenizer.json` | 0.65 | `54682fd945652149…` |
| `models/S-PubMedBert-MedQuAD/tokenizer_config.json` | 0.00 | `e76c62be789fb7fe…` |
| `models/S-PubMedBert-MedQuAD/vocab.txt` | 0.21 | `7b36651908a88bc3…` |
| `models/all-MiniLM-L6-v2/1_Pooling/config.json` | 0.00 | `64230a9762922532…` |
| `models/all-MiniLM-L6-v2/README.md` | 0.01 | `a2521c6638c4ad5e…` |
| `models/all-MiniLM-L6-v2/config.json` | 0.00 | `e68d96b5425efb7e…` |
| `models/all-MiniLM-L6-v2/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/all-MiniLM-L6-v2/model.safetensors` | 86.65 | `1377e9af0ca0b016…` |
| `models/all-MiniLM-L6-v2/modules.json` | 0.00 | `e7989e94b5b809d8…` |
| `models/all-MiniLM-L6-v2/sentence_bert_config.json` | 0.00 | `01d1055f90dbf042…` |
| `models/all-MiniLM-L6-v2/special_tokens_map.json` | 0.00 | `5aa43c2f985a2529…` |
| `models/all-MiniLM-L6-v2/tokenizer.json` | 0.68 | `da0e79933b9ed517…` |
| `models/all-MiniLM-L6-v2/tokenizer_config.json` | 0.00 | `d4f505dfdb84b5a9…` |
| `models/all-MiniLM-L6-v2/vocab.txt` | 0.22 | `07eced375cec144d…` |
| `models/all-mpnet-base-v2/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/all-mpnet-base-v2/README.md` | 0.01 | `ab326e6bb3798d12…` |
| `models/all-mpnet-base-v2/config.json` | 0.00 | `bf07bdf37432b4fc…` |
| `models/all-mpnet-base-v2/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/all-mpnet-base-v2/model.safetensors` | 417.68 | `0b3c8c717335c801…` |
| `models/all-mpnet-base-v2/modules.json` | 0.00 | `e7989e94b5b809d8…` |
| `models/all-mpnet-base-v2/sentence_bert_config.json` | 0.00 | `2b4c737bec5c4c20…` |
| `models/all-mpnet-base-v2/special_tokens_map.json` | 0.00 | `efa8db3ea1576eeb…` |
| `models/all-mpnet-base-v2/tokenizer.json` | 0.68 | `d00dad6f80b7eab8…` |
| `models/all-mpnet-base-v2/tokenizer_config.json` | 0.00 | `0a7ddd0a3eea4e05…` |
| `models/all-mpnet-base-v2/vocab.txt` | 0.22 | `dbd90cb94e2247bd…` |
| `models/e5-base-v2/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/e5-base-v2/README.md` | 0.07 | `a1bb989bba25a4ad…` |
| `models/e5-base-v2/config.json` | 0.00 | `38774b97dbbcbfe9…` |
| `models/e5-base-v2/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/e5-base-v2/model.safetensors` | 417.66 | `35a9ed61a4cf55d5…` |
| `models/e5-base-v2/modules.json` | 0.00 | `e7989e94b5b809d8…` |
| `models/e5-base-v2/sentence_bert_config.json` | 0.00 | `0d36864e76a49566…` |
| `models/e5-base-v2/special_tokens_map.json` | 0.00 | `5aa43c2f985a2529…` |
| `models/e5-base-v2/tokenizer.json` | 0.68 | `d241a60d5e8f04cc…` |
| `models/e5-base-v2/tokenizer_config.json` | 0.00 | `c709b7b834bdb716…` |
| `models/e5-base-v2/vocab.txt` | 0.22 | `07eced375cec144d…` |
| `models/multi-qa-mpnet-base-cos-v1/1_Pooling/config.json` | 0.00 | `fb410fd9c2ed3e3b…` |
| `models/multi-qa-mpnet-base-cos-v1/README.md` | 0.01 | `770e3f5407c86946…` |
| `models/multi-qa-mpnet-base-cos-v1/config.json` | 0.00 | `bf07bdf37432b4fc…` |
| `models/multi-qa-mpnet-base-cos-v1/config_sentence_transformers.json` | 0.00 | `99bfcdccd0ce62cb…` |
| `models/multi-qa-mpnet-base-cos-v1/model.safetensors` | 417.68 | `48eb3d91be5107cd…` |
| `models/multi-qa-mpnet-base-cos-v1/modules.json` | 0.00 | `e7989e94b5b809d8…` |
| `models/multi-qa-mpnet-base-cos-v1/sentence_bert_config.json` | 0.00 | `0d36864e76a49566…` |
| `models/multi-qa-mpnet-base-cos-v1/special_tokens_map.json` | 0.00 | `efa8db3ea1576eeb…` |
| `models/multi-qa-mpnet-base-cos-v1/tokenizer.json` | 0.68 | `b464564756de4c4e…` |
| `models/multi-qa-mpnet-base-cos-v1/tokenizer_config.json` | 0.00 | `2079bbd96955ad29…` |
| `models/multi-qa-mpnet-base-cos-v1/vocab.txt` | 0.22 | `dbd90cb94e2247bd…` |
