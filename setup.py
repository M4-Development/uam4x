from setuptools import setup

setup(
    name='uam4x',
    version='0.2.0',
    packages=['uam4x'],
    author='M4 Engineering',
    author_email='tnascenzi@m4-engineering.com',
    url='http://www.m4-engineering.com',
    include_package_data=True,
    install_requires=[
        'numpy>=2.2.6',
        'pytest>=8.4.2',
        'openmdao>=3.39.0',
        'cpylog',
        'pytest-cov',
        "pytest",
        "pytest-html",
        'pygit2',
        'matplotlib>=3.10.0',
        'openturns==1.26.0',
        'scipy>=1.16.1',
        'tzdata==2025.2',
        'sympy==1.14.0',
        'pandas==2.3.1'

    ],
    description='M4 Engineering developed UAM design space exploration framework with embedded version control',
)
