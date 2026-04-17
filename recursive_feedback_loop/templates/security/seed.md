# Security Review: {{description}}

## Project Context
{{description}}

## Key Source Files
{{source_dirs}}

## Threat Model
{{threat_model}}

## Review Scope
Perform a thorough security review of this project.

For each vulnerability found:
1. Specific file, function, line number
2. Vulnerability class (OWASP category if applicable)
3. Severity (critical/high/medium/low) with justification
4. Proof of concept or attack scenario
5. Concrete fix

Check for:
- Input validation failures (injection, XSS, path traversal)
- Authentication/authorization bypasses
- Data exposure (logging sensitive data, leaking in errors)
- Cryptographic misuse (weak algorithms, hardcoded keys, bad random)
- Race conditions and TOCTOU issues
- Dependency vulnerabilities
- Configuration issues (debug modes, default credentials)
- Memory safety issues (if applicable to the language)

Read the actual source. Don't assume — verify each function handles edge cases correctly.
Do NOT report theoretical issues without evidence in the code.
