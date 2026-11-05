from pathlib import Path

from setuptools import setup


ROOT = Path(__file__).resolve().parent
PUBLIC_PACKAGES = {
    "uam4x_general": "plugins/general/uam4x_general",
    "uam4x_ndarc": "plugins/ndarc/uam4x_ndarc",
    "uam4x_openvsp": "plugins/openvsp/uam4x_openvsp",
    "uam4x_ixgen": "plugins/ixgen/uam4x_ixgen",
}

# Keep non-Python input/reference files in the installed package. Generated
# work directories and Python bytecode are not example source files.
public_package_data = {
    name: [
        str(path.relative_to(ROOT / location))
        for path in (ROOT / location).rglob("*")
        if path.is_file() and path.suffix not in {".py", ".pyc", ".pyo"}
    ]
    for name, location in PUBLIC_PACKAGES.items()
}

setup(
    name="uam4x",
    version="0.4.0",
    packages=["uam4x", *PUBLIC_PACKAGES],
    package_dir={
        "uam4x": "uam4x",
        **PUBLIC_PACKAGES,
    },
    package_data=public_package_data,
    include_package_data=True,
    author="M4 Engineering",
    author_email="tnascenzi@m4-engineering.com",
    url="http://www.m4-engineering.com",
    install_requires=[
        "numpy>=2.2.6",
        "pytest>=8.4.2",
        "openmdao>=3.39.0",
        "cpylog",
        "pytest-cov",
        "pytest",
        "pytest-html",
        "pygit2",
        "matplotlib>=3.10.0",
        "openturns==1.26.0",
        "scipy>=1.16.1",
        "tzdata==2025.2",
        "sympy==1.14.0",
        "pandas==2.3.1",
        "scikit-learn==1.4.2",
    ],
    description="M4 Engineering developed UAM design space exploration framework with embedded version control",
)
