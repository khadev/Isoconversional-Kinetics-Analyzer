# Contributing

Thanks for considering a contribution to **Isoconversional Kinetics
Analyzer**!

## Getting set up

```bash
git clone https://github.com/your-org/isoconversional-kinetics.git
cd isoconversional-kinetics
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

## Running the tests

```bash
# Full suite (GUI smoke tests need a display; use Xvfb if headless)
xvfb-run -a pytest tests/ -v

# Skip GUI tests
pytest tests/ -v --ignore=tests/test_gui_smoke.py

# With coverage
pytest tests/ --cov=isokinetics --cov-report=term-missing
```

## Code style

- Follow PEP 8; keep functions documented with NumPy/Google-style
  docstrings.
- Prefer small, focused functions with type hints.
- No new hard dependencies beyond what's in `requirements.txt` for the
  core library (the whole point of this project is that it works without
  SciPy). If you need SciPy-only functionality, please discuss in an
  issue first.

## Adding a new isoconversional method

1. Implement it in `src/isokinetics/kinetics.py`, following the existing
   signature: `def my_method(df: pd.DataFrame, beta_list: Sequence[float]) -> Dict[float, float]`.
2. Register it in the `METHODS` dict at the bottom of that file.
3. Add it to `METHOD_NAMES` in `src/isokinetics/gui.py` so it shows up as
   a checkbox.
4. Add a validation test in `tests/test_kinetics.py` (ideally reusing the
   `synthetic_dataset` fixture, which has a known ground-truth Eₐ).

## Submitting changes

1. Fork the repo and create a feature branch.
2. Make your changes with tests.
3. Ensure `pytest` passes locally.
4. Open a pull request describing the change and why it's needed.

## Reporting bugs

Please include:
- Python version and OS.
- The exact input file (or a minimal reproducing example) if possible.
- The full traceback / error message.
