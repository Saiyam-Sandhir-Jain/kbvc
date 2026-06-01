# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | ✅ Yes    |

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Please email **security@kbvc.ai** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact

You will receive an acknowledgement within 48 hours and a resolution timeline within 7 days.

## Scope

Key areas of concern:

- **Secrets exposure** — API keys in commit objects, lock files, or log output
- **Path traversal** — `kbvc checkout` or `kbvc ingest` writing outside the repo root
- **Arbitrary code execution** — ingestion pipeline processing malicious content
- **Dependency vulnerabilities** — vulnerabilities in `click`, `pyyaml`, or optional backend packages

## Out of scope

- Vulnerabilities in the vector database or embedding service itself (report to the respective vendor)
- Social engineering
