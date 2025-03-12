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
    "EstimatePZLephareAlgoConfig",
    "EstimatePZLephareAlgoTask",
    "EstimatePZLephareTask",
    "EstimatePZLephareConfig",
]

from rail.estimation.algos.lephare import LephareEstimator
from rail.estimation.estimator import CatEstimator

from .estimate_pz_task import (
    EstimatePZAlgoConfigBase,
    EstimatePZAlgoTask,
    EstimatePZTask,
    EstimatePZTaskConfig,
)


class EstimatePZLephareAlgoConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZLephareAlgoTask

    This will select and configure the LephareEstimator p(z)
    estimation algorithm

    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return LephareEstimator


EstimatePZLephareAlgoConfig._make_fields()


class EstimatePZLephareAlgoTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL Lephare algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_lephare/blob/src/rail/estimation/algos/lephare.py  # noqa
    for algorithm implementation.

    Lephare estimates the p(z) distribution by taking
    a weighted mixture of the nearest neigheboors in
    color space.
    """

    ConfigClass = EstimatePZLephareAlgoConfig
    _DefaultName = "estimatePZLephareAlgo"


class EstimatePZLephareConfig(EstimatePZTaskConfig):
    """Config for EstimatePZLephareTask

    Overrides setDefaults to use Lephare algorithm
    """

    def setDefaults(self) -> None:
        self.pz_algo.retarget(EstimatePZLephareAlgoTask)
        self.pz_algo.stage_name = "lephare"
        self.pz_algo.output_mode = "return"
        self.pz_algo.bands_to_convert = ["u", "g", "r", "i", "z", "y"]
        self.pz_algo.bands = self.pz_algo.get_mag_name_list()
        self.pz_algo.err_bands = self.pz_algo.get_mag_err_name_list()
        self.pz_algo.mag_limits = self.pz_algo.get_mag_lim_dict()
        self.pz_algo.band_a_env = self.pz_algo.get_band_a_env_dict()


class EstimatePZLephareTask(EstimatePZTask):
    """Task that runs RAIL Lephare algorithm for p(z) estimation"""

    ConfigClass = EstimatePZLephareConfig
    _DefaultName = "estimatePZLephare"
