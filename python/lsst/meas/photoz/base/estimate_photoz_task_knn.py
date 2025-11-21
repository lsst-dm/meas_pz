# This file is part of meas_photoz_base.
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
    "EstimatePhotozKNNAlgoConfig",
    "EstimatePhotozKNNAlgoTask",
    "EstimatePhotozKNNConfig",
    "EstimatePhotozKNNTask",
]

from rail.estimation.algos.k_nearneigh import KNearNeighEstimator
from rail.estimation.estimator import CatEstimator

import lsst.pex.config as pexConfig

from .estimate_photoz_task import (
    EstimatePhotozAlgoConfigBase,
    EstimatePhotozAlgoTask,
    EstimatePhotozTask,
    EstimatePhotozTaskConfig,
    photozAlgoRegistry,
)


class EstimatePhotozKNNAlgoConfig(EstimatePhotozAlgoConfigBase):
    """Config for EstimatePhotozKNNAlgoTask.

    This will select and configure the KNearNeighEstimator p(z)
    estimation algorithm.
    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return KNearNeighEstimator

    @classmethod
    def stage_name(cls):
        return "knn"


EstimatePhotozKNNAlgoConfig._make_fields()


@pexConfig.registerConfigurable(EstimatePhotozKNNAlgoConfig.stage_name(), photozAlgoRegistry)
class EstimatePhotozKNNAlgoTask(EstimatePhotozAlgoTask):
    """Subtask to run RAIL KNN algorithm for p(z) estimation.

    See https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/k_nearneigh.py
    for algorithm implementation.
    """

    ConfigClass = EstimatePhotozKNNAlgoConfig
    _DefaultName = "estimatePhotozKNNAlgo"


class EstimatePhotozKNNConfig(EstimatePhotozTaskConfig):
    """Config for EstimatePhotozKNNTask."""

    def setDefaults(self) -> None:
        super().setDefaults()
        name = EstimatePhotozKNNAlgoConfig.stage_name()
        self.connections.algo = name
        self.photoz_algo = name


class EstimatePhotozKNNTask(EstimatePhotozTask):
    """Task that runs RAIL KNN algorithm for p(z) estimation."""

    ConfigClass = EstimatePhotozKNNConfig
    _DefaultName = "estimatePhotozKNN"
