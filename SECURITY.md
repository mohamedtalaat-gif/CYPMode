# Security Policy

## Supported Versions

Pre-1.0: only the latest commit on `main` is supported.

## Reporting a Vulnerability

Please report security issues privately using GitHub's
[Security Advisories](../../security/advisories/new) for this repository
("Report a vulnerability" under the Security tab), rather than opening a
public issue.

## Scope notes specific to this project

- `cypmode/validation/structures.py` parses mmCIF files produced by a local
  Boltz-2 run. It doesn't fetch or execute anything from an untrusted
  source, but a malformed or adversarial CIF file passed to it directly
  (outside the normal Boltz-2 output flow) is unvalidated input — treat it
  the same as any other untrusted file.
- `third_party/` holds full external clones (TDC, Boltz-2) with their own
  git history and license terms, and is intentionally excluded from version
  control — see `.gitignore` and [README.md](README.md) for how to obtain
  them.
