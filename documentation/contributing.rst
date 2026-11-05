Contributing
============

We welcome contributions to uam4x! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/your-username/uam4x.git
      cd uam4x

3. Create a development environment:

   .. code-block:: bash

      python -m venv uam4x_dev
      source uam4x_dev/bin/activate  # On Windows: uam4x_dev\Scripts\activate
      pip install -e .

4. Install documentation dependencies when editing the unified Sphinx source:

   .. code-block:: bash

      python -m pip install -r documentation/requirements.txt
      sphinx-build -M html documentation documentation/_build

Code Style
----------

uam4x follows PEP 8 style guidelines with some modifications:

* Use 4 spaces for indentation
* Maximum line length of 88 characters
* Use type hints for function parameters and return values
* Use Google-style docstrings

Example:

.. code-block:: python

   def calculate_weight(span: float, chord: float, density: float = 1.0) -> float:
       """Calculate component weight.

       Args:
           span: Component span in meters
           chord: Component chord in meters
           density: Material density in kg/m³

       Returns:
           Weight in kg
       """
       return span * chord * density

Testing
-------

All new code must include tests:

1. Unit tests for individual functions
2. Integration tests for component interactions
3. GUI tests for user interface components

Run tests with:

.. code-block:: bash

   pytest tests/

Documentation
-------------

Documentation is automatically generated from docstrings. Follow these guidelines:

1. Write clear, concise docstrings for all public functions and classes
2. Include examples in docstrings where helpful
3. Update this documentation when adding new features
4. Use type hints to improve documentation clarity

Pull Request Process
--------------------

1. Create a feature branch from `main`
2. Make your changes with appropriate tests
3. Update documentation as needed
4. Run the full test suite
5. Submit a pull request with a clear description

Pull Request Guidelines
-----------------------

* Use descriptive commit messages
* Include tests for new functionality
* Update documentation for user-facing changes
* Ensure all tests pass
* Follow the existing code style

Issue Reporting
---------------

When reporting issues:

1. Use the issue template
2. Include steps to reproduce
3. Provide system information
4. Include relevant error messages

Feature Requests
----------------

For feature requests:

1. Check existing issues first
2. Provide a clear description
3. Explain the use case
4. Consider contributing the feature yourself

Code Review
-----------

All code changes require review:

1. Reviewers will check code quality, tests, and documentation
2. Address feedback promptly
3. Be open to suggestions and improvements
4. Keep discussions constructive

Release Process
---------------

Releases are managed by maintainers:

1. Version numbers follow semantic versioning
2. Release notes are generated from pull requests
3. Documentation is updated automatically
4. PyPI packages are built and published

Getting Help
------------

* Check the documentation first
* Search existing issues
* Ask questions in discussions
* Join our community chat

Thank you for contributing to uam4x!
