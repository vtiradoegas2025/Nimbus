# Security Policy

Nimbus is a research simulator, not a network-facing service, so its security
surface is mostly limited to parsing untrusted inputs (configuration YAML,
sounding/hodograph files, and saved run outputs read back by the Studio UI).

## Reporting a Vulnerability

Please report suspected vulnerabilities privately to
victor.tiradoegas2002@gmail.com rather than opening a public issue. Include:

- A description of the issue and its impact
- Steps to reproduce (a minimal config or input file if applicable)
- The affected component (simulator core, Studio UI, or build tooling) and commit

You can expect an initial acknowledgement within a few days. Once a fix is
available, we will coordinate disclosure with you.

## Scope

In scope: memory-safety or input-validation issues in the C++ simulator core and
its parsers, and issues in the Studio desktop app (`ui/`).

Out of scope: denial of service from deliberately pathological simulation inputs
(large grids, extreme parameters), which are expected to be resource-bound.
