# RetailFlow: Intelligent Visual Analytics for Retail Spaces

People counting, zones, dwell time and heatmaps from **angled / side-view CCTV**,
evaluated against public ground truth.

> **Status:** Phases 1–5 (data, pipeline, reliable counting, zones/dwell/heatmaps/staff/blur,
> backend + dashboard) complete. Results so far:
> [`docs/10_RESULTS.md`](docs/10_RESULTS.md). Full README with results, architecture and
> quick start comes in Phase 8. See [`docs/00_PROGRESS_LOG.md`](docs/00_PROGRESS_LOG.md).

## Setup (short)

```powershell
py -3.12 -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
python tools/check_gpu.py
```

Datasets: [`tools/download_data.md`](tools/download_data.md).

## Backend and dashboard

```powershell
docker compose up -d --build        # PostgreSQL + FastAPI (:8000) + Next.js dashboard (:3001)
python run.py --source data/caviar/TwoEnterShop1front/TwoEnterShop1front.mpg --scene scenes/caviar_front.json --api http://127.0.0.1:8000
```

Open <http://localhost:3001>. Details: [`docs/07_BACKEND_DASHBOARD.md`](docs/07_BACKEND_DASHBOARD.md).

## Data acknowledgements

- CAVIAR: EC Funded CAVIAR project/IST 2001 37540 (<https://homepages.inf.ed.ac.uk/rbf/CAVIAR/>), CC BY-SA.
- MOT17 / MOT20: MOTChallenge (<https://motchallenge.net>), CC BY-NC-SA 3.0.
