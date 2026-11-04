.PHONY: setup data train test eda all mlflow-ui serve docker-build docker-run spark-features rag-index copilot-data

# Interpreter used to create the venv (needs 3.10+; macOS's /usr/bin/python3 is 3.9).
# Override like: make setup PYTHON=/path/to/python3.11
PYTHON ?= $(shell command -v python3.11 || echo python3)
VENV_BIN := .venv/bin
# Put src/ on the import path explicitly. On an iCloud-synced Desktop, macOS can flag
# the venv's editable-install .pth file as hidden, and Python then skips it.
export PYTHONPATH := $(CURDIR)/src
# PySpark 4 needs Java 17+. Use the user-level JDK in ~/.local/jdk if JAVA_HOME isn't already set.
JAVA_HOME ?= $(lastword $(wildcard $(HOME)/.local/jdk/jdk-17*/Contents/Home))
export JAVA_HOME

setup:            ## create venv and install everything
	$(PYTHON) -m venv .venv && $(VENV_BIN)/pip install --upgrade pip && $(VENV_BIN)/pip install -r requirements.txt && $(VENV_BIN)/pip install -e .

data:             ## CSV -> SQLite -> order_features table
	$(VENV_BIN)/python -m shipsense.data

train:            ## train all 4 models, write reports/
	$(VENV_BIN)/python -m shipsense.train

test:
	$(VENV_BIN)/pytest -q

eda:              ## rebuild and run the EDA notebook
	$(VENV_BIN)/python scripts/make_eda_notebook.py && $(VENV_BIN)/jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb

all: data train test

mlflow-ui:        ## browse all training runs at http://127.0.0.1:5000
	$(VENV_BIN)/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

serve:            ## run the prediction API at http://127.0.0.1:8000/docs
	$(VENV_BIN)/uvicorn shipsense.api:app --host 127.0.0.1 --port 8000

docker-build:     ## build the API image (run `make train` first so models/ exists)
	docker build -t shipsense-api:latest .

docker-run:       ## run the API container at http://127.0.0.1:8000/docs (Ctrl+C to stop)
	docker run --rm -p 127.0.0.1:8000:8000 --name shipsense-api shipsense-api:latest

spark-features:   ## PySpark feature pipeline; ARGS=--compare checks it row-by-row against SQLite
	$(VENV_BIN)/python spark/order_features_spark.py $(ARGS)

rag-index:        ## chunk + embed knowledge/*.md into a FAISS index (data/processed/rag_index/)
	$(VENV_BIN)/python -m shipsense.rag

copilot-data:     ## build the shipments table + policy index the copilot needs (after `make data`)
	$(VENV_BIN)/python -m shipsense.shipments && $(VENV_BIN)/python -m shipsense.rag
