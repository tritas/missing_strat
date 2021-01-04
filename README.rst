==============================
Python Missing Data Strategies
==============================


.. image:: https://img.shields.io/pypi/v/missing_strat.svg
        :target: https://pypi.python.org/pypi/missing_strat

.. image:: https://img.shields.io/travis/tritas/missing_strat.svg
        :target: https://travis-ci.org/tritas/missing_strat

.. image:: https://readthedocs.org/projects/missing-strat/badge/?version=latest
        :target: https://missing-strat.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status


.. image:: https://pyup.io/repos/github/tritas/missing_strat/shield.svg
     :target: https://pyup.io/repos/github/tritas/missing_strat/
     :alt: Updates



Missing strat is a python library that pre-processes your dataset to fill missing data.

The idea is to be able to easily state your assumption about the missingness mechanism,
specify the outputs your expect (best-effort, force removal of missingness) and implement
the inference process as fast and effiently as possible.

Essentially an intelligent automl mechanism with user-provided specification of the
underlying causal mechanism (if known) and budget in the form of time constraint. We do
not want the pre-processing step to run for ages if it only brings incremental
improvement to the downstream task.

Easily plugging this transform in ML pipelines such as sklean or TFX is important (we do
not address the could scenario ATM). The idea is to operate with a scikit-learn-like
class interface, such that the code is widely portable in the pydata ecosystem.
Recognizing that many new tools are part of the ML ecosystem, we provide a modular
execution backend class that should allow running on: numpy, dask, spark udf, stan,
tensorflow, pytorch. Contributions welcome!


* Free software: BSD license
* Documentation: https://missing-strat.readthedocs.io.


Features
--------

* TODO

Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
