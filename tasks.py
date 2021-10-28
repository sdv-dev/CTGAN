import os
import re
import shutil
import stat
from pathlib import Path

from invoke import task


@task
def check_dependencies(c):
    c.run('python -m pip check')


@task
def unit(c):
    c.run('python -m pytest ./tests/unit --cov=ctgan --cov-report=xml')


@task
def integration(c):
    c.run('python -m pytest ./tests/integration')


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


@task
def install_minimum(c):
    with open('setup.py', 'r') as setup_py:
        lines = setup_py.read().splitlines()

    versions = []
    started = False
    for line in lines:
        if started:
            if line == ']':
                break

            line = line.strip()
            line = re.sub(r',?<=?[\d.]*,?', '', line)
            line = re.sub(r'>=?', '==', line)
            line = re.sub(r"""['",]""", '', line)
            versions.append(line)

        elif line.startswith('install_requires = ['):
            started = True

    c.run(f'python -m pip install {" ".join(versions)}')


@task
def minimum(c):
    check_dependencies(c)
    install_minimum(c)
    unit(c)
    end_to_end(c)
    numerical(c)


@task
def lint(c):
    check_dependencies(c)
    c.run('flake8 ctgan')
    c.run('flake8 tests --ignore=D,SFS2')
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
