# Security Policy

## Supported Versions

Only the latest release receives security updates.

## Reporting a Vulnerability

Do not open a public issue. Send a private report via:

- GitHub Security Advisories (use the Report a vulnerability button on the repo)
- Email: clocekrindo@yandex.ru

You should get a response within 48 hours.

## Best Practices

- Never commit API keys or credentials to the repo
- Use .env files for local config (see .env.example)
- Keep deps updated with pip install -r requirements.txt --upgrade
- Use docker-compose.prod.yml for deployments
