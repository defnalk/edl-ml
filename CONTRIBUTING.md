# Contributing to edl-ml

Thank you for considering a contribution! This project welcomes bug reports,
feature requests, documentation improvements, and pull requests.

## Development setup

```bash
git clone https://github.com/defnalk/edl-ml
cd edl-ml
make install-dev       # editable install + pre-commit hooks
```

The `dev` extra pulls in `ruff`, `mypy`, `pytest`, `hypothesis`, `nox`,
`pre-commit`, `pytest-cov`.

## Commit style

Please use [Conventional Commits](https://www.conventionalcommits.org/):

| Prefix       | Meaning                                           |
|--------------|---------------------------------------------------|
| `feat:`      | A new user-facing feature                         |
| `fix:`       | A bug fix                                         |
| `refactor:`  | Code restructuring with no behavioural change     |
| `test:`      | Adding or correcting tests                        |
| `docs:`      | Documentation only                                |
| `chore:`     | Infrastructure, dependencies, build system        |
| `ci:`        | CI configuration                                  |
| `perf:`      | Performance improvement                           |

## Checks

Every PR must pass:

```bash
make lint          # ruff check + format
make typecheck     # mypy strict
make test          # pytest with hypothesis
make test-cov      # coverage report
make docs          # mkdocs --strict
```

## Adding a physics feature

When extending the physics core please also add:

1. A closed-form or textbook-reference validation test (exact tolerance).
2. At least one Hypothesis property test asserting a physical invariant.
3. A docstring section explaining the equation and assumptions.

## Adding an ML feature

For new model architectures, training tricks, or evaluation metrics:

1. Ensure the feature is reproducible under a fixed seed.
2. Add a smoke test training a tiny version of the model.
3. Add an entry to the appropriate MkDocs page.

## Reporting issues

Please include:

- `python --version`, `pip freeze` for the relevant environment.
- Minimal reproducer (ideally one-file).
- Full traceback.

## Code of Conduct

Be kind, assume good faith, and write code you would be happy to see
reviewed by a future version of yourself.
