# Release workflow

The process of releasing a new version involves several steps:

1. [Install CTGAN from source](#install-ctgan-from-source)

2. [Linting and tests](#linting-and-tests)

3. [Make a release candidate](#make-a-release-candidate)

4. [Integration with SDV](#integration-with-sdv)

5. [Milestone](#milestone)

6. [Update HISTORY](#update-history)

7. [Check the release](#check-the-release)

8. [Update stable branch and bump version](#update-stable-branch-and-bump-version)

9. [Create the Release on GitHub](#create-the-release-on-github)

10. [Close milestone and create new milestone](#close-milestone-and-create-new-milestone)

11. [Release on Conda-Forge](#release-on-conda-forge)

## Install CTGAN from source

Clone the project and install the development requirements before starting the release process. Alternatively, with your virtualenv activated:

```bash
git clone https://github.com/sdv-dev/CTGAN.git
cd CTGAN
git checkout main
make install-develop
```

## Linting and tests

Execute the tests and linting. The tests must end with no errors:

```bash
make test && make lint
```

And you will see something like this:

```
Coverage XML written to file ./integration_cov.xml

======================= 24 passed, 7 warnings in 51.23s ========================
....
invoke lint
No broken requirements found.
All checks passed!
28 files already formatted
```

The execution has finished with no errors, 0 test skipped and 166 warnings.

## Make a release candidate

1. On the CTGAN GitHub page, navigate to the [Actions][actions] tab.
2. Select the `Release` action.
3. Run it on the main branch. Make sure `Release candidate` is checked and `Test PyPI` is not.
4. Check on [PyPI][ctgan-pypi] to assure the release candidate was successfully uploaded.
  - You should see X.Y.ZdevN PRE-RELEASE

[actions]: https://github.com/sdv-dev/CTGAN/actions
[ctgan-pypi]: https://pypi.org/project/CTGAN/#history

## Integration with SDV

### Create a branch on SDV to test the candidate

Before doing the actual release, we need to test that the candidate works with SDV. To do this, we can create a branch on SDV that points to the release candidate we just created using the following steps:

1. Create a new branch on the SDV repository.

```bash
git checkout -b test-ctgan-X.Y.Z
```

2. Update the pyproject.toml to set the minimum version of CTGAN to be the same as the version of the release. For example,

```toml
'ctgan>=X.Y.Z.dev0'
```

3. Push this branch. This should trigger all the tests to run.

```bash
git push --set-upstream origin test-ctgan-X.Y.Z
```

4. Check the [Actions][sdv-actions] tab on SDV to make sure all the tests pass.

[sdv-actions]: https://github.com/sdv-dev/SDV/actions

## Milestone

It's important to check that the GitHub and milestone issues are up to date with the release.

You neet to check that:

- The milestone for the current release exists.
- All the issues closed since the latest release are associated to the milestone. If they are not, associate them.
- All the issues associated to the milestone are closed. If there are open issues but the milestone needs to
  be released anyway, move them to the next milestone.
- All the issues in the milestone are assigned to at least one person.
- All the pull requests closed since the latest release are associated to an issue. If necessary, create issues
  and assign them to the milestone. Also assign the person who opened the issue to them.

## Update HISTORY
Run the [Release Prep](https://github.com/sdv-dev/CTGAN/actions/workflows/prepare_release.yml) workflow. This workflow will create a pull request with updates to HISTORY.md

Make sure HISTORY.md is updated with the issues of the milestone:

```
# History

## X.Y.Z (YYYY-MM-DD)

### New Features

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/CTGAN/issues/<issue>) by @resolver

### General Improvements

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/CTGAN/issues/<issue>) by @resolver

### Bug Fixed

* <ISSUE TITLE> - [Issue #<issue>](https://github.com/sdv-dev/CTGAN/issues/<issue>) by @resolver
```

The issue list per milestone can be found [here][milestones].

[milestones]: https://github.com/sdv-dev/CTGAN/milestones

Put the pull request up for review and get 2 approvals to merge into `main`.

## Check the release
Once HISTORY.md has been updated on `main`, check if the release can be made:

```bash
make check-release
```

## Update stable branch and bump version
The `stable` branch needs to be updated with the changes from `main` and the version needs to be bumped.
Depending on the type of release, run one of the following:

* `make release`: This will release the version that has already been bumped (patch, minor, or major). By default, this is typically a patch release. Use this when the changes are bugfixes or enhancements that do not modify the existing user API. Changes that modify the user API to add new features but that do not modify the usage of the previous features can also be released as a patch.
* `make release-minor`: This will bump and release the next minor version. Use this if the changes modify the existing user API in any way, even if it is backwards compatible. Minor backwards incompatible changes can also be released as minor versions while the library is still in beta state. After the major version v1.0.0 has been released, minor version can only be used to add backwards compatible API changes.
* `make release-major`: This will bump and release the next major version. Use this if the changes modify the user API in a backwards incompatible way after the major version v1.0.0 has been released.

Running one of these will **push commits directly** to `main`.
At the end, you should see the 3 commits on `main` (from oldest to newest):
- `make release-tag: Merge branch 'main' into stable`
- `Bump version: X.Y.Z.devN → X.Y.Z`
- `Bump version: X.Y.Z -> X.Y.A.dev0`

## Create the Release on GitHub

After the update to HISTORY.md is merged into `main` and the version is bumped, it is time to [create the release GitHub](https://github.com/sdv-dev/CTGAN/releases/new).
- Create a new tag with the version number with a v prefix (e.g. v0.3.1)
- The target should be the `stable` branch
- Release title is the same as the tag (e.g. v0.3.1)
- This is not a pre-release (`Set as a pre-release` should be unchecked)

Click `Publish release`, which will kickoff the release workflow and automatically upload the package to [public PyPI](https://pypi.org/project/ctgan/).

## Close milestone and create new milestone

Finaly, **close the milestone** and, if it does not exist, **create the next milestone**.

## Release on conda-forge

After the release is published on [public PyPI](https://pypi.org/project/ctgan/), Anacanoda will automatically open a [PR on conda-forge](https://github.com/conda-forge/ctgan-feedstock/pulls). Make sure the dependencies match and then merge the PR for the anaconda release to be published.
