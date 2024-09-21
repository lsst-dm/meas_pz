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
    "EstimatePZKNNTask",
    "EstimatePZKNNConfig",
    
]

from rail.estimation.estimator import CatEstimator
from rail.estimation.algos.k_nearneigh import KNearNeighEstimator

from .estimate_pz_task import (
    EstimatePZAlgoConfigBase,
    EstimatePZAlgoTask,
    EstimatePZTaskConfig,
    EstimatePZTask,
)


class EstimatePZKNNAlgoConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZAlgoKNNTask

    This will select and configure the KNearNeighEstimator p(z)
    estimation algorithm

    See https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/k_nearneigh.py  # noqa
    for parameters and default values.
    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return KNearNeighEstimator


EstimatePZKNNAlgoConfig._make_fields()


class EstimatePZKNNAlgoTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL KNN algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/k_nearneigh.py  # noqa
    for algorithm implementation.

    KNN estimates the p(z) distribution by taking
    a weighted mixture of the nearest neigheboors in
    color space.
    """

    ConfigClass = EstimatePZKNNAlgoConfig
    _DefaultName = "estimatePZKNNAlgo"


class EstimatePZKNNConfig(EstimatePZTaskConfig):
    """Config for EstimatePZKNNTask
    
    Overrides setDefaults to use KNN algorithm
    """

    def setDefaults(self):
        self.pz_algo.retarget(EstimatePZKNNAlgoTask)
        self.pz_algo.stage_name='knn'
        self.pz_algo.output_mode='return'
        self.pz_algo.bands=['mag_g_lsst','mag_r_lsst','mag_i_lsst','mag_z_lsst','mag_y_lsst']
        self.pz_algo.ref_band='mag_i_lsst'
        self.pz_algo.band_a_env=dict(g=3.64,r=2.70,i=2.06,z=1.58,y=1.31)


class EstimatePZKNNTask(EstimatePZTask):
    """ Task that runs RAIL KNN algorithm for p(z) estimation """
    
    ConfigClass = EstimatePZKNNConfig
    _DefaultName = "estimatePZKNN"

    
