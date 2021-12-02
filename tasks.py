import glob
import operator
import os
import re
import platform
import shutil
import stat
from pathlib import Path

from invoke import task

COMPARISONS = {
    '>=': operator.ge,
    '>': operator.gt,
    '<': operator.lt,
    '<=': operator.le
}


@task
def check_dependencies(c):
    c.run('python -m pip check')


@task
def unit(c):
    c.run('python -m pytest ./tests/unit --cov=ctgan --cov-report=xml')


@task
def integration(c):
    c.run('python -m pytest ./tests/integration --reruns 3')


@task
def readme(c):
    test_path = Path('tests/readme_test')
    if test_path.exists() and test_path.is_dir():
        shutil.rmtree(test_path)

    cwd = os.getcwd()
    os.makedirs(test_path, exist_ok=True)
    shutil.copy('README.md', test_path / 'README.md')
    os.chdir(test_path)
    c.run('rundoc run --single-session python3 -t python3 README.md')
    os.chdir(cwd)
    shutil.rmtree(test_path)


def _validate_python_version(line):
    python_version_match = re.search(r"python_version(<=?|>=?)\'(\d\.?)+\'", line)
    if python_version_match:
        python_version = python_version_match.group(0)
        comparison = re.search(r'(>=?|<=?)', python_version).group(0)
        version_number = python_version.split(comparison)[-1].replace("'", "")
        comparison_function = COMPARISONS[comparison]
        return comparison_function(platform.python_version(), version_number)

    return True


@task
def install_minimum(c):
    with open('setup.py', 'r') as setup_py:
        lines = setup_py.read().splitlines()

    versions = []
    started = False
    for line in lines:
        if started:
            if line == ']':
                started = False
                continue

            line = line.strip()
            if _validate_python_version(line):
                requirement = re.match(r'[^>]*', line).group(0)
                requirement = re.sub(r"""['",]""", '', requirement)
                version = re.search(r'>=?[^(,|#)]*', line).group(0)
                if version:
                    version = re.sub(r'>=?', '==', version)
                    version = re.sub(r"""['",]""", '', version)
                    requirement += version

                versions.append(requirement)

        elif (line.startswith('install_requires = [') or
             line.startswith('pomegranate_requires = [')):
            started = True

    c.run(f'python -m pip install {" ".join(versions)}')


@task
def minimum(c):
    install_minimum(c)
    check_dependencies(c)
    unit(c)
    integration(c)


@task
def lint(c):
    check_dependencies(c)
    c.run('flake8 ctgan')
    c.run('flake8 tests --ignore=D101')
    c.run('isort -c --recursive ctgan tests')


def remove_readonly(func, path, _):
    "Clear the readonly bit and reattempt the removal"
    os.chmod(path, stat.S_IWRITE)
    func(path)


@task
def rmdir(c, path):
    try:
        shutil.rmtree(path, onerror=remove_readonly)
    except PermissionError:
        pass
