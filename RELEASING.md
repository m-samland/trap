# Releasing trap

trap is published on PyPI as [`trap-hci`](https://pypi.org/p/trap-hci) by
[`.github/workflows/publish.yml`](.github/workflows/publish.yml).
It uses Trusted Publishing: PyPI trusts that workflow file together with the GitHub
environment `pypi`, so no API token is stored anywhere.

## Steps

1. On `develop`, rename the CHANGELOG's `[Unreleased]` heading to the new version and
   date, for example `## [2.1.0] - 2026-10-15`, and merge that by PR.
   CI on `develop` must be green, including the end-to-end test.
2. Optional dry run: run the Publish workflow by hand
   (`gh workflow run publish.yml --ref develop`), which uploads a dev version to TestPyPI.
   Install it without taking dependencies from TestPyPI:
   `pip download --no-deps --index-url https://test.pypi.org/simple/ trap-hci==<version>`,
   then `pip install` the downloaded wheel, so its dependencies come from PyPI.
3. Merge `develop` into `main` by PR.
4. On GitHub, draft a new release: a new tag `v2.1.0` targeting `main`, title `v2.1.0`,
   and the CHANGELOG section as notes.
   Publishing the release starts the Publish workflow.
5. The workflow builds, checks that the wheel's version matches the tag, and then waits
   for approval of the `pypi` environment.
   Approve it on the workflow run's page (Actions → Publish → *Review deployments*).
6. In a clean environment, `pip install trap-hci==2.1.0`, check that `import trap`
   works, and run `pytest -m e2e` from a checkout of the tag against the installed package.
7. Merge `main` back into `develop`, so that development versions count from the new tag.
8. Update the packages that depend on trap, such as spherical's `pipeline` extra.

## Things to know

- A version can be uploaded only once, even after deleting it from PyPI.
  If a release is broken, fix it with a new patch version.
- A GitHub release marked as a pre-release (for example `v2.2.0rc1`) is published to
  PyPI as well; pip installs it only when asked for pre-releases.
- Renaming `publish.yml` or the environments `pypi` and `testpypi`, or moving the
  repository, breaks publishing until the trusted publishers on pypi.org and
  test.pypi.org are updated to match.
