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
    "EstimatePZTPZAlgoConfig",
    "EstimatePZTPZAlgoTask",
    "EstimatePZTPZTask",
    "EstimatePZTPZConfig",
]

from rail.estimation.algos.tpz_lite import TPZliteEstimator
from rail.estimation.estimator import CatEstimator

from .estimate_pz_task import (
    EstimatePZAlgoConfigBase,
    EstimatePZAlgoTask,
    EstimatePZTask,
    EstimatePZTaskConfig,
)


class EstimatePZTPZAlgoConfig(EstimatePZAlgoConfigBase):
    """Config for EstimatePZTPZAlgoTask

    This will select and configure the TPZliteEstimator p(z)
    estimation algorithm

    See https://github.com/LSSTDESC/rail_tpz/blob/src/rail/estimation/algos/tpz_lite.py  # noqa
    for parameters and default values.
    """

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return TPZliteEstimator


EstimatePZTPZAlgoConfig._make_fields()


class EstimatePZTPZAlgoTask(EstimatePZAlgoTask):
    """SubTask that runs RAIL TPZ algorithm for p(z) estimation

    See https://github.com/LSSTDESC/rail_tpz/blob/src/rail/estimation/algos/tpz_lite.py  # noqa
    for parameters and default values.
    """

    ConfigClass = EstimatePZTPZAlgoConfig
    _DefaultName = "estimatePZTPZAlgo"


class EstimatePZTPZConfig(EstimatePZTaskConfig):
    """Config for EstimatePZTPZTask

    Overrides setDefaults to use TPZ algorithm
    """

    def setDefaults(self) -> None:
        self.pz_algo.retarget(EstimatePZTPZAlgoTask)
        self.pz_algo.stage_name = "tpz"
        self.pz_algo.output_mode = "return"
        self.pz_algo.bands_to_convert = ["u", "g", "r", "i", "z", "y"]
        self.pz_algo.mag_limits = self.pz_algo.get_mag_lim_dict()
        self.pz_algo.band_a_env = self.pz_algo.get_band_a_env_dict()


class EstimatePZTPZTask(EstimatePZTask):
    """Task that runs RAIL TPZ algorithm for p(z) estimation"""

    ConfigClass = EstimatePZTPZConfig
    _DefaultName = "estimatePZTPZ"
