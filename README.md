# Dangote Stock Prediction

Project layout:

- data/raw/ - raw CSV files
- data/processed/ - cleaned/processed datasets
- src/ - source code (subpackages: `ingest`, `clean`, `train`, `visualization`, `features`, `evaluation`; entrypoint: `src/main.py`)
- models/ - saved models (.joblib)
- figures/ - generated figures (PDF)

How to run the pipeline:

   - From the repository root: `python run.py`
   - Or directly: `python src/main.py`