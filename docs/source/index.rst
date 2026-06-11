EV-QA-Framework Documentation
==============================

ML-powered QA framework for EV battery systems. Already includes 948 passing tests, ~93% coverage, Docker Compose, Prometheus metrics, and compliance tests for UN 38.3, IEC 62660, SAE J2464, ISO 12405, GB/T 31484, GB/T 31486, GB 38031.

.. toctree::
   :maxdepth: 3
   :caption: Contents:

   getting_started
   modules
   api/index
   examples/index

Quick start
-----------

Clone and inspect locally:

.. code-block:: bash

   git clone https://github.com/remontsuri/EV-QA-Framework.git
   cd EV-QA-Framework
   docker compose up -d
   open http://localhost:8081/coverage/

Or run the CLI directly:

.. code-block:: bash

   uv run pytest -v
   uv run python run_factory_inspection.py

Overview
--------

This framework validates BMS telemetry and predicts fault conditions before they become failures. It covers:

- telementry validation
- anomaly detection
- thermal runaway prediction
- SOH prediction
- cell imbalance analysis
- CAN/J1939 emulation and DBC parsing
- fleet analytics
- digital twin
- V2G scenarios
- HIL integration
- compliance testing against international standards
- observability via Prometheus metrics and Grafana-ready output

For architecture and use cases, start with :doc:`getting_started`.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
