.. py:currentmodule:: lsst.meas.photoz.base

.. _lsst.meas.photoz.base:

#####################
lsst.meas.photoz.base
#####################

meas_photoz_base contains classes and tasks for running photometric redshift estimation algorithms on LSST catalogs.

meas_photoz_base relies on the DESC `Redshift Assessment Infrastructure Layers (RAIL) <https://github.com/LSSTDESC/rail/>`_ framework to provide implementations of algorithms and to parameterize/quantize probability distributions.

Currently, meas_photoz_base only provides tasks for a limited number of algorithms with prerequisite packages available in the `lsst_distrib` environment.
More are available from `meas_photoz_algorithms <https://github.com/lsst-dm/meas_photoz_algorithms>`_.

.. _lsst.meas.photoz.base-using:

Using lsst.meas.photoz.base
===========================

Consult the unit tests and/or the pipelines in `drp_pipe <https://github.com/lsst/drp_pipe>`_ for suggested usage.

To populate the `photozAlgoRegistry`, import from `lsst.meas.photoz.base.all_algos`.

.. toctree linking to topics related to using the module's APIs.

.. .. toctree::
..    :maxdepth: 1

.. _lsst.meas.photoz.base-contributing:

Contributing
============

``lsst.meas.photoz.base`` is developed at https://github.com/lsst/meas_photoz_base.
You can find Jira issues for this module under the `meas_photoz_base <https://rubinobs.atlassian.net/issues/?jql=component%20%3D%20meas_photoz_base>`_ component.


.. If there are topics related to developing this module (rather than using it), link to this from a toctree placed here.

.. .. toctree::
..    :maxdepth: 1

.. .. _lsst.meas.photoz.base-scripts:

.. Script reference
.. ================

.. .. TODO: Add an item to this toctree for each script reference topic in the scripts subdirectory.

.. .. toctree::
..    :maxdepth: 1

.. .. _lsst.meas.photoz.base-pyapi:

Python API reference
====================

.. automodapi:: lsst.meas.photoz.base
   :no-main-docstr:
   :no-inheritance-diagram:
