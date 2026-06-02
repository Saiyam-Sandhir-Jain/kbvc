# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| 0.1.x | ✅ Active |

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please report security issues by emailing:  
**saiyam.sandhir.jain@gmail.com**

Include:
- A description of the vulnerability
- Steps to reproduce
- Potential impact
- Your suggested fix (optional)

You will receive a response within 72 hours. We take security seriously and will work with you to address the issue promptly.

## Security Notes

- **`kbvc.lock` never contains API keys** — it only records provider names and model identifiers. API keys are stored only in `.kbvc/config`, which should be in `.gitignore`.
- **`.kbvc/config` should always be in `.gitignore`** — it contains API keys. `kbvc init` adds it automatically.
- KBVC does not transmit any data except to the embedding and vector DB providers you explicitly configure.
