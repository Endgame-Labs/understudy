# Security policy

## Reporting a vulnerability

Report vulnerabilities privately through GitHub: open the repository's **Security** tab and choose
**Report a vulnerability**. Do not open a public issue, pull request, or discussion for a suspected vulnerability.

Include the affected version or commit, the steps to reproduce, and the impact. Reports are acknowledged, and fixes are
coordinated with the reporter before public disclosure.

## Supported versions

Understudy is pre-1.0. Fixes are made on `main` and ship in the next release.

## Deployment

`understudy serve` sees every prompt and state it proxies, writes capture logs and snapshots to disk, and has no listener
authentication. [docs/security.md](docs/security.md) describes these properties and how to deploy around them.
