# Moteur multimodal de recherche et d'assistance (texte + image + audio)

Projet 3 · INF50101 · ENSPY. Recherche texte → image, image → texte et audio → contenu dans un espace
commun de 512 dimensions (CLIP, AST + tête MLP), fusion tardive des scores, index FAISS exact,
application Flask, évaluation Recall@K / Precision@K.

## 1. Installation (Python 3.10 ou plus)

```bash
python -m venv .venv && source .venv/bin/activate     # Windows : .venv\Scripts\activate
make install                                           # pip install -e . puis requirements.txt
make test                                              # tests sans GPU ni modèles
```

## 2. Démonstration immédiate (sans données ni modèles)

```bash
make fake                                              # artefacts factices + index + SQLite
MMSEARCH_LOAD_MODELS=0 make app                        # http://127.0.0.1:5000 (pages, embeddings, filtres)
```

Sans modèles les onglets de recherche sont désactivés ; la page Embeddings fonctionne (PCA).

## 3. Exécution complète avec les vrais datasets

1. Télécharger Flickr8k, Clotho v2 et ESC-50 dans `data/raw/` (arborescence dans
   `src/mmsearch/data/ingest.py`). ESC-50 sert à la classification audio zero-shot ; sans ses fichiers,
   cette classification est désactivée, mais la recherche audio sur Clotho reste disponible.
2. `make data` : écrit `data/processed/manifest.csv` et `captions.csv`.
3. `make embed` : extraction des embeddings (GPU T4 recommandé : Colab / Kaggle, reprise automatique).
4. `make train` : tête audio (`artifacts/models/audio_proj.pt`), projection ESC-50 et fusion précoce.
5. `make fusion` : règle les poids de la fusion tardive ; reporter `best_w_mod` dans `config.yaml`.
6. `make index` : construit `artifacts/index/*.faiss` et `collection.db`.
7. `make projections` : PCA, t-SNE et UMAP précalculés pour la page Embeddings.
8. `make eval` puis `make figures` : tableaux (`results_tables.tex`), figures, études de cas, biais.
9. `make app` : application sur http://127.0.0.1:5000 (`MMSEARCH_LOAD_MODELS=1`, par défaut).

L'encodage d'un clip audio envoyé par l'utilisateur prend environ 5 s sur CPU (clip tronqué à 10 s dans
la démo) ; activer `device: cuda` dans `config.yaml` si une machine GPU est disponible.

Pour préparer uniquement la recherche audio de la plateforme sans recalculer les embeddings Flickr8k
ni les embeddings ESC-50 d'évaluation, lancer `make embed-audio-app`. Les classes ESC-50 utilisées par
la classification des requêtes sont lues depuis `esc50.csv`; l'encodeur AST et son checkpoint audio
restent nécessaires.

## 4. API

| Route | Méthode | Rôle |
|---|---|---|
| `/api/search` | POST | `kind` (text, image, audio), `q` ou `file`, `k`, `mode` (fused, modal), `filters` (JSON) |
| `/api/item/<id>` | GET | métadonnées et légendes d'un item |
| `/media/<id>` | GET | fichier image ou audio |
| `/api/embeddings` | GET | points 2D (`method` pca, tsne, umap ; `dataset`, `modality`, `limit`) |
| `/api/classify` | POST | classification audio zero-shot (ESC-50) |
| `/api/health` | GET | état et modalités disponibles |

Erreurs : `{"error": {"code": ..., "message": ...}}` avec les statuts 400, 404, 413, 415 ou 503.

## 5. Structure

`src/mmsearch/` (data, encoders, training, fusion, index, retrieval, tasks, viz, eval), `app/` (Flask),
`scripts/`, `tests/`, `config.yaml`, `Makefile`. Les artefacts (`artifacts/`) ne sont pas versionnés.

## 6. Dépannage

| Symptôme | Piste |
|---|---|
| `FileNotFoundError` sur un `.npy` | l'étape précédente (`make embed`, `make train`) n'a pas été lancée |
| Onglet audio grisé | `artifacts/models/audio_proj.pt` absent : `make train` |
| Classification audio indisponible | Ajouter ESC-50 dans `data/raw/esc50/audio/` et `data/raw/esc50/meta/esc50.csv` ; la recherche audio Clotho n'en dépend pas |
| 503 `projection_unavailable` | `make projections` (t-SNE et UMAP sont précalculés ; PCA se calcule à la demande) |
| Mémoire GPU insuffisante | réduire `--chunk` dans `extract_embeddings.py` ou utiliser `--fp16` |
