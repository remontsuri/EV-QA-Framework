# Contributing to EV-QA-Framework

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Code of Conduct

- Be respectful and constructive
- Follow PEP 8 style guide
- Write clear, descriptive commit messages
- Test your changes before submitting

## Getting Started

1. **Fork the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/EV-QA-Framework.git
   cd EV-QA-Framework
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

## Development Workflow

### Making Changes

1. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes

3. Run tests:
   ```bash
   pytest -v
   pytest --cov=ev_qa_framework
   ```

4. Commit with descriptive message:
   ```bash
   git commit -m "feat: Add new anomaly detection algorithm"
   ```

5. Push and create Pull Request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all functions
- Use meaningful variable names

## Testing Requirements

- All new features must include tests
- Maintain or improve code coverage
- Tests must pass before merging

## Pull Request Process

1. Update documentation
2. Ensure tests pass
3. Request review from maintainers
4. Address feedback
5. Merge after approval

## Areas for Contribution

- Battery management algorithms
- Test coverage improvements
- Documentation enhancements
- Performance optimizations
- Bug fixes
- CI/CD improvements

## Questions?

Open an issue or discussion for questions!

---

**Thank you for contributing!** 🉋
