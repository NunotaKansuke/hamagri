# Development

Install an editable development version and run the checks:

```bash
python -m pip install -e ".[all,dev]"
pytest -q
ruff check src tests examples
```

`examples/build_and_plot.py` is a small direct-VBMicrolensing integration
example. For older Linux systems, see
[VBMicrolensing compatibility](vbm-compatibility.md).
