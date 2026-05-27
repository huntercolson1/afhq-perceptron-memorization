# Run Log

Date: 2026-05-27

Machine: local macOS Python 3.12 virtual environment under `.venv`

## Dataset

Exact Kaggle handle:

```text
andrewmvd/animal-faces
```

Download command used:

```bash
.venv/bin/python scripts/download_afhq.py --source kagglehub
```

KaggleHub `1.0.1` had an import mismatch with `kagglesdk` in this environment, so the project pins `kagglehub==0.3.13`. With that pin, the exact Kaggle dataset downloaded successfully and extracted to:

```text
data/raw/kaggle_animal_faces
```

Observed extracted file count:

```text
16130 files
```

## Experiment Run

Command:

```bash
.venv/bin/python scripts/run_experiment.py \
  --data-root data/raw/kaggle_animal_faces \
  --sample-sizes 500 1000 2000 4096 4097 5000 \
  --max-epochs 50 \
  --output-dir outputs
```

Results:

| N | rank(X with bias) | VC construction train error | Perceptron train error | Perceptron converged? |
|---:|---:|---:|---:|:---|
| 500 | 500 | 0.0000 | 0.0000 | yes |
| 1000 | 1000 | 0.0000 | 0.0340 | no |
| 2000 | 2000 | 0.0000 | 0.1585 | no |
| 4096 | 4096 | 0.0000 | 0.2925 | no |
| 4097 | 4097 | 0.0000 | 0.2563 | no |
| 5000 | 4097 | 0.0262 | 0.2252 | no |

## Verification

Commands run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/run_experiment.py --synthetic --sample-sizes 20 40 80 --max-epochs 3 --output-dir outputs/smoke
.venv/bin/python -m jupyter nbconvert --to notebook --execute notebooks/01_afhq_vc_perceptron_demo.ipynb --output 01_afhq_vc_perceptron_demo.executed.ipynb --output-dir notebooks --ExecutePreprocessor.timeout=1200
.venv/bin/python - <<'PY'
import nbformat
from pathlib import Path
path = Path('notebooks/01_afhq_vc_perceptron_demo.ipynb')
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
print(f'validated {path} with {len(nb.cells)} cells')
PY
```

Observed verification:

```text
3 passed in 4.94s
validated notebooks/01_afhq_vc_perceptron_demo.ipynb with 17 cells
```

The temporary smoke output folder was removed after verification so the durable outputs reflect the AFHQ run.
