# Contributing to EV-QA-Framework

Thanks for considering contributing.

## How to Contribute

### Report Bugs

Open an issue at github.com/remontsuri/EV-QA-Framework/issues. Include your Python version, OS, the full error traceback, and steps to reproduce.

### Suggest Features

Open a feature request issue. Describe the problem you're trying to solve and how you'd solve it.

### Submit Code

```bash
git clone https://github.com/YOUR_USERNAME/EV-QA-Framework.git
cd EV-QA-Framework
git checkout -b feature/your-feature-name
pip install -r requirements-dev.txt

# Run tests before committing
pytest -v --cov=ev_qa_framework

# Format code
black ev_qa_framework tests

git commit -m "feat: add your feature description"
git push origin feature/your-feature-name
```

Open a pull request targeting the `main` branch. Keep PRs focused on one thing.

## Development Guidelines

- Code style: Black (default config), isort, flake8
- Tests: pytest with pytest-asyncio for async code
- Documentation in English, clear docstrings for public APIs
- Runtime deps go in requirements.txt, dev-only in requirements-dev.txt

## Commit Convention

Use conventional commits: `feat:`, `fix:`, `docs:`, `test:`, `refactor:`, `perf:`, `docker:`, `ci:`
