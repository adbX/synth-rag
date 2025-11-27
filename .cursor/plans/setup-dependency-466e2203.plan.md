<!-- 466e2203-1dd4-40a5-9256-3f3cf7705a03 e3006774-1ba5-43e9-b8a7-49f45ba0ee83 -->
# Setup Dependency Groups for Documentation

## Changes to Make

### 1. Update [`pyproject.toml`](pyproject.toml)

- Remove `mkdocs-material>=9.7.0` and `mkdocstrings-python>=2.0.0` from the main `dependencies` array
- Add a new `[dependency-groups]` section at the end of the file (before `[build-system]`)
- Create a `docs` group containing:
  - `mkdocs-material>=9.7.0`
  - `mkdocstrings-python>=2.0.0`

This will keep all runtime dependencies in the main array while isolating documentation-specific dependencies.

### 2. Update [`.github/workflows/ci.yml`](.github/workflows/ci.yml)

- Change the "Install dependencies" step from `uv sync` to `uv sync --only-group docs`
- This will install only the docs group dependencies plus the project itself (which mkdocstrings needs to document the code)

## UV Commands Reference

After these changes, here are the key commands you'll use:

```bash
# Install only docs dependencies (for CI/building docs)
uv sync --only-group docs

# Install all default dependencies (for development)
uv sync

# Install default dependencies + docs group
uv sync --group docs

# Install multiple groups
uv sync --group docs --group dev
```

## Benefits

- **Faster CI builds**: Only installs ~2 packages instead of ~20
- **Cleaner separation**: Documentation dependencies are clearly separated from runtime dependencies
- **Flexibility**: Easy to add more groups later (e.g., `dev`, `test`, `notebook`)