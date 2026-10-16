<!-- Thanks for contributing! Keep changes test-first and behavior-preserving where possible. -->

## Summary

What does this PR change and why?

## Related issue

Closes #

## Changes

-
-

## Checklist

- [ ] Tests added/updated for the change (TDD: a failing test first)
- [ ] `make lint` passes (`ruff check .`, `ruff format --check .`, `pyright`)
- [ ] `make test-fast` and `make test-emulator` pass locally
- [ ] Behavior-preserving changes leave `tests/golden/` untouched; a golden that must change is its own commit with the reason
- [ ] The dependency rule is respected (domain imports nothing from the package; infrastructure never imports application or cli)
- [ ] Docs/README updated if behavior or commands changed
