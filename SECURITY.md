# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 2.0.x   | Active |
| 1.1.x   | Security fixes only |
| 1.0.x   | End of life |
| < 1.0   | End of life |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue in EV-QA-Framework, please report it responsibly.

**Do not open a public GitHub issue for security vulnerabilities.**

Instead, choose one of the following:

1. **Email** (preferred): Send a detailed report to the maintainers. Use the subject line: `[SECURITY] EV-QA-Framework Vulnerability Report`.

2. **GitHub Private Vulnerability Reporting**: Use [GitHub's private vulnerability reporting](https://github.com/remontsuri/EV-QA-Framework/security/advisories/new).

### What to Include

- Description of the vulnerability.
- Affected component (e.g., `can_bus.py`, `dbc_parser.py`, `cli.py`).
- Affected version(s).
- Steps to reproduce or proof-of-concept code.
- Impact (data exposure, code execution, denial of service, etc.).
- Suggested fix (optional).

### Response Timeline

- Acknowledgment: within 48 hours.
- Initial assessment: within 5 business days.
- Fix and disclosure: coordinated with the reporter; typically within 30 days for critical issues.

### Disclosure Policy

- We follow responsible disclosure practices.
- The reporter will be credited in the advisory (unless they request anonymity).
- A GitHub Security Advisory will be published after the fix is released.
- CVE identifiers will be requested for qualifying vulnerabilities.

## Security Considerations

### CAN Bus Module

The `can_bus.py` module can interact with physical CAN hardware.

- Do not connect to vehicle CAN buses without proper authorization and safety precautions.
- Simulation mode (default) does not require physical hardware and is safe for development.
- When using `python-can` with physical interfaces, ensure the interface is properly isolated.

### DBC Parser

The `dbc_parser.py` module parses untrusted input files.

- Validate file size before parsing (recommended limit: 10 MB).
- The parser does not execute arbitrary code, but malformed files may cause excessive memory usage.

### CLI and Dashboard

- The CLI and dashboard accept file paths and network interfaces as arguments.
- Do not expose the dashboard to untrusted networks without authentication.
- The Prometheus metrics endpoint (`/metrics`) exposes system metrics — restrict access in production.

### Dependencies

- Dependencies are pinned in `uv.lock` for reproducible builds.
- Run `uv audit` to check for known vulnerabilities in dependencies.
- Keep dependencies updated: `uv lock --upgrade`.

## Hardening Recommendations

1. Run tests in an isolated environment: `uv run pytest -v`.
2. Use simulation mode for CAN bus during development.
3. Enable ruff's security linting rules: `uv run ruff check --select S .`.
4. Review third-party DBC files before parsing.
5. Do not run the dashboard on `0.0.0.0` in production without a reverse proxy.

## Known Limitations

- The framework does not implement authentication or authorization. It is a QA/testing tool, not a production service.
- Telemetry data is processed in-memory only; no persistent storage is implemented.
- The HIL module (`hil.py`) interfaces with physical hardware. Follow all applicable safety standards when using this module.
