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
    "EstimatePZTrainZTask",
    "EstimatePZTrainZConfig",
]

import numpy as np

from pandas import DataFrame
from rail.estimation.algos.train_z import TrainZEstimator
from .estimate_pz_task import EstimatePZAlgoConfigBase, EstimatePZAlgoTask


class EstimatePZTrainZConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZTrainZTask

    This will select and comnfigure the TrainZEsimator p(z)
    estimation algorithm

    See https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/train_z.py  # noqa
    for parameters and default values.
    """

    estimator_class = TrainZEstimator


EstimatePZTrainZConfig._make_fields()


class EstimatePZTrainZTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL TrainZ algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_base/blob/main/src/rail/estimation/algos/train_z.py  # noqa
    for algorithm implementation.

    TrainZ is just a placeholder algorithm that assigns that same
    p(z) distribution (taken from the input model file) to every object.
    """

    ConfigClass = EstimatePZTrainZConfig

    def _get_mags_and_errs(
        self,
        fluxes: DataFrame,
        mag_offset: float,
    ) -> dict[str, np.array]:
        return dict(lsst_i_mag=fluxes["i_gaap1p0Flux"])
