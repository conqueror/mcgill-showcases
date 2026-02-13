# Troubleshooting

## `uv: command not found`
Install `uv` and reopen your terminal.

## Dependency resolution fails
From project directory:
```bash
uv lock
uv sync --extra dev
```

## Kaggle auth error (causal project)
Check:
```bash
ls -l ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

## Notebook doesn't open or validate
Reinstall notebook dependencies and validate JSON format:
```bash
python -m pip install nbformat
```

## Tests fail only on one project
Run commands inside that project to isolate:
```bash
cd projects/<project-name>
make check
```
