# This file is part of meas_pz
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


import numpy as np
import pytest

from lsst.meas.pz.estimate_pz_task import EstimatePZAlgoConfigBase, EstimatePZAlgoTask


@pytest.fixture(scope="module")
def fluxes():
    """Return logarithmically-spaced fluxes."""
    return 10 ** np.linspace(0, 4, 100)


@pytest.fixture(scope="module")
def flux_errors(fluxes):
    """Return sqrt(N) errors on fluxes."""
    return np.sqrt(fluxes)


@pytest.fixture(scope="module")
def magnitudes(fluxes):
    """Return AB mags of fluxes."""
    return -2.5 * np.log10(fluxes) + 31.4


@pytest.fixture(scope="module")
def magnitude_errors(fluxes, flux_errors):
    """Return magnitude errors from fluxes."""
    return flux_errors / (fluxes * 0.4 * np.log(10))


def test_flux_to_mag(fluxes, magnitudes):
    """Test flux to magnitude conversions."""
    mags_convert = EstimatePZAlgoTask._flux_to_mag(fluxes, 31.4, np.nan)
    assert np.allclose(mags_convert, magnitudes, atol=1e-10, rtol=1e-12)


def test_flux_err_to_mag_err(fluxes, flux_errors, magnitude_errors):
    """Test flux error to magnitude error conversions."""
    mags_convert = EstimatePZAlgoTask._flux_err_to_mag_err(fluxes, flux_errors, mag_conv=np.log(10) * 0.4)
    assert np.allclose(mags_convert, magnitude_errors, atol=1e-10, rtol=1e-12)


def test_algo_config():
    """Test default initialization of base config class."""
    config = EstimatePZAlgoConfigBase(stage_name="test")
    config.validate()
