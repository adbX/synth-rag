<!-- ad4ed2d3-2e1e-4d25-b5aa-a2bef9bd7dca ba4c32be-9645-496f-a32e-12534cd541c8 -->
# Clone Computo Repos Script

## Implementation Steps

1. **Add PyGithub dependency** to `pyproject.toml`

2. **Implement `clone_computo_repos.py`** with the following functionality:

- Use PyGithub to list all repos from `computorg` organization (no auth)
- Filter repos starting with "published-"
- Parse repo names to extract:
- `date`: YYYYMM format → "YYYY-MM" (e.g., "202510" → "2025-10")
- `metadata`: everything after the date (e.g., "durand-fast")
- Load existing `documents/computo/computo_repos.csv` if it exists
- Skip already cloned repos (check CSV)
- Clone new repos using `subprocess.run(["git", "clone", repo_url, target_path])`
- Show progress with tqdm progress bar
- Save/update CSV with columns: repo_name, repo_url, date, metadata
- Use pathlib for all file operations
- Create `documents/computo/` directory if needed

3. **Script structure**:

- Function to parse repo name → extract date and metadata
- Function to load existing CSV
- Function to clone a single repo
- Main function to orchestrate the process

## Key Files

- `src/qdrant_init/clone_computo_repos.py`: The main script
- `pyproject.toml`: Add PyGithub dependency
- `documents/computo/computo_repos.csv`: Output tracking file (created by script)

### To-dos

- [ ] Add PyGithub to pyproject.toml dependencies
- [ ] Implement clone_computo_repos.py with all required functionality