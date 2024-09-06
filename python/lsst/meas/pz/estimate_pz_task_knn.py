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
    "EstimatePZKNNTask",
    "EstimatePZKNNConfig",
]


from rail.estimation.algos.k_nearneigh import KNearNeighEstimator

from .estimate_pz_task import EstimatePZAlgoConfigBase, EstimatePZAlgoTask


class EstimatePZKNNConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZKNNTask

    This will select and comnfigure the KNearNeighEstimator p(z)
    estimation algorithm

    See https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/k_nearneigh.py  # noqa
    for parameters and default values.
    """

    @property
    def estimator_class(self) -> type[CatEstimator]:
        return KNearNeighEstimator


EstimatePZKNNConfig._make_fields()


class EstimatePZKNNTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL KNN algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_sklearn/blob/main/src/rail/estimation/algos/k_nearneigh.py  # noqa
    for algorithm implementation.

    KNN estimates the p(z) distribution by taking
    a weighted mixture of the nearest neigheboors in
    color space.
    """

    ConfigClass = EstimatePZKNNConfig
    _DefaultName = "estimatePZKNN"
