# Contributing

Thanks for contributing to `mcgill-showcases`.

## Contribution Principles
- Keep changes focused (one clear improvement per PR).
- Keep explanations student-friendly.
- Update docs when behavior or outputs change.
- Add or update tests for code changes.

## Local Setup
```bash
make sync
```

## Local Quality Gate
```bash
make check
```

## Pull Request Checklist
- [ ] Scope is clear and limited.
- [ ] `make check` passes.
- [ ] User-facing docs are updated.
- [ ] New notebooks/docs render without broken links.
- [ ] Added/changed commands are included in README.

## Commit Style
Use Conventional Commits when possible:
- `feat: ...`
- `fix: ...`
- `docs: ...`
- `test: ...`
- `chore: ...`

## Student Experience Guardrails
- Prefer clear wording over jargon.
- Explain "why this matters" for each new concept.
- Include at least one decision-oriented interpretation for major outputs.
