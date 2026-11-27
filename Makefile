# Cibles principales (voir README.md). Exécuter depuis la racine du dépôt.
PY ?= python

.PHONY: install fake test lint data embed embed-audio-app train index fusion projections eval figures bias app report slides clean

install:
	pip install -e . && pip install -r requirements.txt

fake:                       ## artefacts factices + index (développement sans données)
	$(PY) scripts/make_fake_artifacts.py --n-img 200 --n-aud 100

test:
	$(PY) -m pytest -q

lint:
	ruff check src scripts tests app

data:                       ## manifest.csv, captions.csv (datasets déjà téléchargés dans data/raw)
	$(PY) -m mmsearch.data.ingest

embed:                      ## extraction des embeddings (GPU recommandé)
	$(PY) scripts/extract_embeddings.py --stage text  --dataset flickr8k
	$(PY) scripts/extract_embeddings.py --stage text  --dataset flickr8k --encoder sbert
	$(PY) scripts/extract_embeddings.py --stage text  --dataset clotho
	$(PY) scripts/extract_embeddings.py --stage image --dataset flickr8k --fp16
	$(PY) scripts/extract_embeddings.py --stage audio_raw --dataset clotho
	$(PY) scripts/extract_embeddings.py --stage audio_raw --dataset esc50

embed-audio-app:            ## pipeline audio pour la plateforme (Clotho, sans embeddings ESC-50 de test)
	$(PY) -m mmsearch.data.ingest
	$(PY) scripts/extract_embeddings.py --stage text --dataset clotho
	$(PY) scripts/extract_embeddings.py --stage audio_raw --dataset clotho
	$(PY) -m mmsearch.training.train_audio_proj
	$(PY) -m mmsearch.index.build_index

train:                      ## tête audio et fusion précoce
	$(PY) -m mmsearch.training.train_audio_proj
	$(PY) scripts/extract_embeddings.py --stage audio_proj --dataset esc50
	$(PY) -m mmsearch.training.train_early_fusion

fusion:                     ## poids de la fusion tardive (à reporter dans config.yaml)
	$(PY) -m mmsearch.fusion.tune_weights

index:
	$(PY) -m mmsearch.index.build_index

projections:
	$(PY) -m mmsearch.viz.projections

eval:                       ## tableaux, résultats, études de cas, zero-shot, biais
	$(PY) -m mmsearch.eval.run_eval --split test
	$(PY) -m mmsearch.tasks.zero_shot_cls
	$(PY) -m mmsearch.eval.bias_analysis

figures:
	$(PY) -m mmsearch.eval.make_figures

report:                     ## rapport PDF (pdflatex + bibtex)
	cd report && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

slides:                     ## présentation Beamer (deux passes)
	cd slides && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

app:
	flask --app app run --host 127.0.0.1 --port 5000

clean:
	rm -rf .pytest_cache .ruff_cache artifacts/figures artifacts/projections
	cd report && rm -f *.aux *.log *.bbl *.blg *.toc *.out main.pdf
	cd slides && rm -f *.aux *.log *.nav *.out *.snm *.toc main.pdf
