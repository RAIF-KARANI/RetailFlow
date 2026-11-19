# RetailFlow: Intelligent Visual Analytics for Retail Spaces

People counting, zones, dwell time and heatmaps from **angled / side-view CCTV**,
evaluated against public ground truth.

> **Status:** Phase 1 (setup and data) complete. Full README with results, architecture and
> quick start comes in Phase 8. See [`docs/00_PROGRESS_LOG.md`](docs/00_PROGRESS_LOG.md).

## Setup (short)

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python tools/check_gpu.py
```

Datasets: [`tools/download_data.md`](tools/download_data.md).

## Data acknowledgements

- CAVIAR: EC Funded CAVIAR project/IST 2001 37540 (<https://homepages.inf.ed.ac.uk/rbf/CAVIAR/>), CC BY-SA.
- MOT17 / MOT20: MOTChallenge (<https://motchallenge.net>), CC BY-NC-SA 3.0.
