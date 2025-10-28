# This file is part of meas.photoz.base.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = [
    "EstimatePhotozTrainZAlgoConfig",
    "EstimatePhotozTrainZAlgoTask",
    "EstimatePhotozTrainZConfig",
    "EstimatePhotozTrainZTask",
]

import numpy as np
from astropy.table import Table
from rail.estimation.algos.train_z import TrainZEstimator
from rail.estimation.estimator import CatEstimator

import lsst.pex.config as pexConfig

from .estimate_photoz_task import (
    EstimatePhotozAlgoConfigBase,
    EstimatePhotozAlgoTask,
    EstimatePhotozTask,
    EstimatePhotozTaskConfig,
    photozAlgoRegistry,
)


class EstimatePhotozTrainZAlgoConfig(EstimatePhotozAlgoConfigBase):
    """Config for EstimatePhotozTrainZAlgoTask."""

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return TrainZEstimator

    @classmethod
    def stage_name(cls):
        return "trainz"

    def setDefaults(self):
        self.band_a_env = {"i": 2.06}


EstimatePhotozTrainZAlgoConfig._make_fields()


@pexConfig.registerConfigurable(EstimatePhotozTrainZAlgoConfig.stage_name(), photozAlgoRegistry)
class EstimatePhotozTrainZAlgoTask(EstimatePhotozAlgoTask):
    """Subtask to run RAIL TrainZ algorithm for p(z) estimation.

    See https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/train_z.py
    for algorithm implementation.

    TrainZ is just a placeholder algorithm that assigns that same
    p(z) distribution (taken from the input model file) to every object.
    """

    ConfigClass = EstimatePhotozTrainZAlgoConfig
    _DefaultName = "estimatePhotozTrainZAlgo"

    def _get_mags_and_errs(
        self,
        fluxes: Table,
        mag_offset: float,
    ) -> dict[str, np.ndarray]:
        flux_names = self.config.get_flux_names()
        mag_names = self.config.get_mag_names()

        mag_dict = {}
        # loop over bands, make mags and mag errors and fill dict
        for band, band_name in flux_names.items():
            fluxVals = fluxes[band_name]
            mag_dict[mag_names[band]] = self._flux_to_mag(
                fluxVals,
                mag_offset,
                99.0,
            )
        return mag_dict


class EstimatePhotozTrainZConfig(EstimatePhotozTaskConfig):
    """Config for EstimatePhotozTrainZTask."""

    def setDefaults(self) -> None:
        super().setDefaults()
        name = EstimatePhotozTrainZAlgoConfig.stage_name()
        self.connections.algo = name
        self.photoz_algo = name


class EstimatePhotozTrainZTask(EstimatePhotozTask):
    """Task to run RAIL TrainZ algorithm for p(z) estimation."""

    ConfigClass = EstimatePhotozTrainZConfig
    _DefaultName = "estimatePhotozTrainZ"
