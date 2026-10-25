PY ?= .venv/bin/python

.PHONY: setup data all pipeline test dashboard powerbi sql examples clean

setup:            ## create venv + install deps
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

data:             ## download the public datasets (UCI news, TweetEval)
	$(PY) -m sei.download

all: data pipeline test   ## everything: data -> analysis -> Power BI export -> tests

pipeline:
	$(PY) -m sei.pipeline

test:
	$(PY) -m pytest -q tests

sql:              ## interactive SQL prompt over every dataset
	$(PY) -m sei.sql

examples:         ## run the 14 example queries
	$(PY) -m sei.sql --file queries/examples.sql

dashboard:
	.venv/bin/streamlit run dashboard/app.py

powerbi:          ## re-export Power BI files (reads CSVs from GitHub); DATA_FOLDER=C:\path\ for a local-file model
	$(PY) -m sei.powerbi $(if $(DATA_FOLDER),--source local --data-folder "$(DATA_FOLDER)",)

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet reports/figures/*.png reports/tables/*.csv
