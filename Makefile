PY ?= .venv/bin/python

.PHONY: setup setup-latest fixture data all pipeline test dashboard powerbi sql examples clean

setup:            ## venv with pinned versions (Python 3.12)
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements-lock.txt

setup-latest:     ## venv with the latest allowed versions
	python3 -m venv .venv && .venv/bin/pip install -U pip && .venv/bin/pip install -r requirements.txt

fixture:          ## pipeline + tests on a 3,000-user fixture
	SEI_WORKDIR=$${TMPDIR:-/tmp}/sei-fixture SEI_N_USERS=3000 $(PY) -m sei.pipeline && \
	SEI_WORKDIR=$${TMPDIR:-/tmp}/sei-fixture SEI_N_USERS=3000 $(PY) -m pytest -q tests

data:             ## download UCI news and TweetEval
	$(PY) -m sei.download

all: data pipeline test   ## data, pipeline, tests

pipeline:
	$(PY) -m sei.pipeline

test:
	$(PY) -m pytest -q tests

sql:              ## SQL prompt over every dataset
	$(PY) -m sei.sql

examples:         ## run queries/examples.sql
	$(PY) -m sei.sql --file queries/examples.sql

dashboard:
	.venv/bin/streamlit run dashboard/app.py

powerbi:          ## re-export Power BI files; DATA_FOLDER=C:\path\ for local CSVs
	$(PY) -m sei.powerbi $(if $(DATA_FOLDER),--source local --data-folder "$(DATA_FOLDER)",)

clean:
	rm -rf data/raw/*.parquet data/processed/*.parquet reports/figures/*.png reports/tables/*.csv
