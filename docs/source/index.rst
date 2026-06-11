EV-QA-Framework Documentation
==============================

ML-powered QA Framework for Electric Vehicle & IoT Battery Testing.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   modules
   api/index

Overview
--------

EV-QA-Framework provides a comprehensive toolkit for battery quality assurance
in electric vehicle applications. It covers:

- **State of Health (SOH) prediction** — ML models for battery degradation analysis
- **Battery scoring** — Automated quality scoring and grading
- **Physics-based features** — Domain-specific feature engineering
- **Fleet analytics** — Fleet-wide battery health monitoring
- **Digital twin** — Battery simulation and emulation
- **V2G scenarios** — Vehicle-to-grid testing scenarios
- **HIL testing** — Hardware-in-the-loop test automation
- **Test standards** — GB/T and international standard compliance

Installation
------------

.. code-block:: bash

   uv pip install -e .

For development with docs support:

.. code-block:: bash

   uv pip install -e ".[dev,docs]"

Quick Start
-----------

.. code-block:: python

   from ev_qa_framework.soh_predictor import SOHPredictor
   from ev_qa_framework.battery_scoring import BatteryScorer

   predictor = SOHPredictor()
   scorer = BatteryScorer()

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
