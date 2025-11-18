################
meas_photoz_base
################

``meas_photoz_base`` is a package in the `LSST Science Pipelines <https://pipelines.lsst.io>`_.

meas_photoz_base contains classes and tasks for running photometric redshift estimation algorithms on LSST catalogs.

meas_photoz_base relies on the DESC `Redshift Assessment Infrastructure Layers (RAIL) <https://github.com/LSSTDESC/rail/>`_ framework to provide implementations of algorithms and to parameterize/quantize probability distributions.

Currently, meas_photoz_base only provides tasks for a limited number of algorithms with prerequisite packages available in the lsst_distrib environment.
More are available from `meas_photoz_algorithms <https://github.com/lsst-dm/meas_photoz_algorithms>`_.
