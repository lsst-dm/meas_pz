# This file is part of meas_pz.
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
    "EstimatePZTrainZAlgoConfig",
    "EstimatePZTrainZAlgoTask",
    "EstimatePZTrainZTask",
    "EstimatePZTrainZConfig",
]

import numpy as np
import astropy.table as atable
from rail.estimation.estimator import CatEstimator
from rail.estimation.algos.train_z import TrainZEstimator

from .estimate_pz_task import (
    EstimatePZAlgoConfigBase,
    EstimatePZAlgoTask,
    EstimatePZTaskConfig,
    EstimatePZTask,
)


class EstimatePZTrainZAlgoConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZTrainZAlgoTask

    This will select and comnfigure the TrainZEsimator p(z)
    estimation algorithm

    See https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/train_z.py  # noqa
    for parameters and default values.
    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return TrainZEstimator


EstimatePZTrainZAlgoConfig._make_fields()


class EstimatePZTrainZAlgoTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL TrainZ algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/train_z.py  # noqa
    for algorithm implementation.

    TrainZ is just a placeholder algorithm that assigns that same
    p(z) distribution (taken from the input model file) to every object.
    """

    ConfigClass = EstimatePZTrainZAlgoConfig
    _DefaultName = "estimatePZTrainZAlgo"

    def _get_mags_and_errs(
        self,
        fluxes: atable,
        mag_offset: float,
    ) -> dict[str, np.array]:

        flux_names = self._get_flux_names()
        mag_names = self._get_mag_names()

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


class EstimatePZTrainZConfig(EstimatePZTaskConfig):
    """Config for EstimatePZTrainZTask
    
    Overrides setDefaults to use TrainZ algorithm
    """

    def setDefaults(self):
        self.pz_algo.retarget(EstimatePZTrainZAlgoTask)
        self.pz_algo.stage_name='trainz'
        self.pz_algo.output_mode='return'
        self.pz_algo.band_a_env=dict(i=2.06)


class EstimatePZTrainZTask(EstimatePZTask):
    """ Task that runs RAIL TrainZ algorithm for p(z) estimation """
    
    ConfigClass = EstimatePZTrainZConfig
    _DefaultName = "estimatePZTrainZ"

    
