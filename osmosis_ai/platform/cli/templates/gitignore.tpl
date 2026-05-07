# ── Python ───────────────────────────────────────────
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
dist/
build/
*.egg

# ── Virtual environments ─────────────────────────────
.venv/
venv/
env/

# ── IDE ──────────────────────────────────────────────
.idea/
.vscode/
*.swp
*.swo
*~

# ── Testing / Linting ────────────────────────────────
.pytest_cache/
.mypy_cache/
.ruff_cache/
htmlcov/
.coverage
coverage.xml

# ── Environment / secrets ────────────────────────────
.env
.env.*

# ── Agent tooling (personal overrides only) ──────────
# `.claude/settings.json` is committed — it registers the osmosis plugin
# marketplace for the whole team. Personal overrides go in settings.local.json.
.claude/settings.local.json

# ── Osmosis local state ──────────────────────────────
.osmosis/**
!.osmosis/project.toml

# ── OS ──────────────────────────────────────────────
.DS_Store
Thumbs.db
