# Documentation moved

The single Sphinx source for the framework, public plugins, and in-repository
examples is [`../../documentation`](../../documentation). From this README's
directory (`uam4x/plugins/documentation`), run:

```bash
python -m pip install -r ../../documentation/requirements.txt
sphinx-build -M html ../../documentation ../../documentation/_build
```

Open `../../documentation/_build/html/index.html`.
