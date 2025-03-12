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
    "EstimatePZGPZAlgoConfig",
    "EstimatePZGPZAlgoTask",
    "EstimatePZGPZTask",
    "EstimatePZGPZConfig",
]

from rail.estimation.algos.gpz import GPzEstimator
from rail.estimation.estimator import CatEstimator

from .estimate_pz_task import (
    EstimatePZAlgoConfigBase,
    EstimatePZAlgoTask,
    EstimatePZTask,
    EstimatePZTaskConfig,
)


class EstimatePZGPZAlgoConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZGPZAlgoTask

    This will select and configure the GPzEstimator p(z)
    estimation algorithm

    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return GPzEstimator


EstimatePZGPZAlgoConfig._make_fields()


class EstimatePZGPZAlgoTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL GPZ algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_gpz_v1/blob/src/rail/estimation/algos/gpz.py  # noqa
    for algorithm implementation.

    """

    ConfigClass = EstimatePZGPZAlgoConfig
    _DefaultName = "estimatePZGPZAlgo"


class EstimatePZGPZConfig(EstimatePZTaskConfig):
    """Config for EstimatePZGPZTask

    Overrides setDefaults to use GPZ algorithm
    """

    def setDefaults(self) -> None:
        self.pz_algo.retarget(EstimatePZGPZAlgoTask)
        self.pz_algo.stage_name = "gpz"
        self.pz_algo.output_mode = "return"
        self.pz_algo.bands_to_convert = ["u", "g", "r", "i", "z", "y"]
        self.pz_algo.ref_band = self.pz_algo.mag_template.format(band='i')
        self.pz_algo.bands = self.pz_algo.get_mag_name_list()
        self.pz_algo.err_bands = self.pz_algo.get_mag_err_name_list()
        self.pz_algo.mag_limits = self.pz_algo.get_mag_lim_dict()
        self.pz_algo.band_a_env = self.pz_algo.get_band_a_env_dict()
        self.pz_algo.replace_error_vals = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]


class EstimatePZGPZTask(EstimatePZTask):
    """Task that runs RAIL GPZ algorithm for p(z) estimation"""

    ConfigClass = EstimatePZGPZConfig
    _DefaultName = "estimatePZGPZ"
