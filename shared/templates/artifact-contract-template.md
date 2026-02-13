# Artifact Contract Template

Each project should include `artifacts/manifest.json` with:

```json
{
  "version": 1,
  "required_files": [
    "artifacts/..."
  ]
}
```

Validation command:

```bash
make verify
```

