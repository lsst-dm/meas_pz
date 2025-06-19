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
    "EstimatePZKNNAlgoConfig",
    "EstimatePZKNNAlgoTask",
    "EstimatePZKNNConfig",
    "EstimatePZKNNTask",
]

from rail.estimation.algos.k_nearneigh import KNearNeighEstimator
from rail.estimation.estimator import CatEstimator

from .estimate_pz_task import (
    EstimatePZAlgoConfigBase,
    EstimatePZAlgoTask,
    EstimatePZTask,
    EstimatePZTaskConfig,
)


class EstimatePZKNNAlgoConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZKNNAlgoTask.

    This will select and configure the KNearNeighEstimator p(z)
    estimation algorithm.
    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return KNearNeighEstimator


EstimatePZKNNAlgoConfig._make_fields()


class EstimatePZKNNAlgoTask(EstimatePZAlgoTask):
    """Subtask to run RAIL KNN algorithm for p(z) estimation.

    See https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/k_nearneigh.py
    for algorithm implementation.
    """

    ConfigClass = EstimatePZKNNAlgoConfig
    _DefaultName = "estimatePZKNNAlgo"


class EstimatePZKNNConfig(EstimatePZTaskConfig):
    """Config for EstimatePZKNNTask."""

    def setDefaults(self) -> None:
        self.pz_algo.retarget(EstimatePZKNNAlgoTask)
        self.pz_algo.stage_name = "knn"
        self.pz_algo.output_mode = "return"
        self.pz_algo.bands_to_convert = ["u", "g", "r", "i", "z", "y"]
        self.pz_algo.ref_band = self.pz_algo.mag_template.format(band="i")
        self.pz_algo.bands = self.pz_algo.get_mag_name_list()
        self.pz_algo.mag_limits = self.pz_algo.get_mag_lim_dict()
        self.pz_algo.band_a_env = self.pz_algo.get_band_a_env_dict()
        self.pz_algo.id_col = "objectId"
        self.pz_algo.calc_summary_stats = True


class EstimatePZKNNTask(EstimatePZTask):
    """Task that runs RAIL KNN algorithm for p(z) estimation."""

    ConfigClass = EstimatePZKNNConfig
    _DefaultName = "estimatePZKNN"
