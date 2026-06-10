# Contributing to EV-QA-Framework

## Code of Conduct

- Be respectful and constructive.
- Follow PEP 8 style guide.
- Write clear, descriptive commit messages.
- Test your changes before submitting.

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/EV-QA-Framework.git
   cd EV-QA-Framework
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync --all-extras
   ```
   This creates a virtual environment and installs all dependencies (including dev and ML extras) in one step.

3. **Install pre-commit hooks**
   ```bash
   uv run pre-commit install --hook-type pre-commit --hook-type pre-push
   ```
   This installs two hooks:
   - `pre-commit` — runs ruff lint + format on staged files before each commit
   - `pre-push` — runs full test suite before each push to remote

   To run manually without committing:
   ```
   uv run pre-commit run --all-files
   ```

4. **Activate the virtual environment** (optional — `uv run` handles this automatically)
   ```bash
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes.

3. Run linting:
   ```bash
   uv run ruff check .
   uv run ruff format --check .
   ```

4. Run type checking:
   ```bash
   uv run mypy ev_qa_framework/
   ```

5. Run tests:
   ```bash
   uv run pytest -v
   uv run pytest --cov=ev_qa_framework --cov-report=term-missing
   ```

6. Commit with descriptive message (conventional commits):
   ```bash
   git commit -m "feat: add new anomaly detection algorithm"
   git commit -m "fix: correct SOH scaler serialization round-trip"
   git commit -m "docs: update fleet analytics examples"
   ```

7. Push and create Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8.
- Use type hints where possible.
- Write docstrings for all public functions and classes (Google style).
- Use meaningful variable names.
- Maximum line length: 100 characters (enforced by ruff).

## Testing Requirements

- All new features must include tests.
- Maintain or improve code coverage (currently 86%).
- Tests must pass before merging.
- Use `pytest` with `pytest-cov` for coverage.
- Place tests in the `tests/` directory, mirroring the source structure:
  ```
  tests/
    test_framework.py
    test_models.py
    test_analysis.py
    test_battery_scoring.py
    test_fleet_analytics.py
    ...
  ```

## Pull Request Process

1. Update documentation (README.md, docstrings, examples).
2. Ensure all tests pass: `uv run pytest -v`.
3. Ensure linting passes: `uv run ruff check .`.
4. Request review from maintainers.
5. Address feedback.
6. Merge after approval.

## Areas for Contribution

- Battery management algorithms.
- Test coverage improvements.
- Documentation enhancements.
- Performance optimizations.
- Bug fixes.
- CI/CD improvements.
- New battery chemistry definitions.
- Additional V2G scenarios.
- Hardware-in-the-Loop driver support.
- Compliance testing for additional standards.

## Project Structure

```
ev_qa_framework/          # 22 modules — see README.md for full list
tests/                    # 592 tests
dashboard/                # FastAPI + Grafana
examples/                 # Example CSV telemetry and DBC files
docs/                     # Additional documentation
pyproject.toml            # Project config, dependencies, build system
uv.lock                   # Reproducible dependency lockfile
```

## Questions?

Open an issue or discussion.
