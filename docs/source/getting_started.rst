Getting Started
===============

Installation
------------

Install the package using ``uv``:

.. code-block:: bash

   uv pip install -e "."

Optional dependencies:

.. code-block:: bash

   # ML features (TensorFlow)
   uv pip install -e ".[ml]"

   # Hardware interfaces (CAN bus, serial)
   uv pip install -e ".[hardware]"

   # Development tools (ruff, mypy, pre-commit)
   uv pip install -e ".[dev]"

   # Documentation build
   uv pip install -e ".[docs]"

Running Tests
-------------

.. code-block:: bash

   uv run pytest                  # all tests
   uv run pytest --cov            # with coverage
   uv run pytest tests/test_soh_predictor.py  # single module

Building Documentation
----------------------

.. code-block:: bash

   cd docs
   make html

Output is written to ``docs/build/html/``.

Configuration
-------------

The framework uses built-in defaults from ev_qa_framework/config.py. See config.yaml for customization.
Key configuration sections:

- ``battery`` — cell chemistry, capacity, voltage limits
- ``soh`` — prediction model parameters
- ``fleet`` — fleet analytics thresholds
- ``v2g`` — V2G scenario definitions
