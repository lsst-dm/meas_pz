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

from rail.estimation.algos.bpz_lite import BPZliteEstimator
from rail.estimation.estimator import CatEstimator

__all__ = [
    "EstimatePhotozBPZAlgoConfig",
    "EstimatePhotozBPZAlgoTask",
    "EstimatePhotozBPZConfig",
    "EstimatePhotozBPZTask",
]

from .estimate_photoz_task import (
    EstimatePhotozAlgoConfigBase,
    EstimatePhotozAlgoTask,
    EstimatePhotozTask,
    EstimatePhotozTaskConfig,
)


class EstimatePhotozBPZAlgoConfig(EstimatePhotozAlgoConfigBase):
    """Config for EstimatePhotozBPZAlgoTask."""

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return BPZliteEstimator


EstimatePhotozBPZAlgoConfig._make_fields()


class EstimatePhotozBPZAlgoTask(EstimatePhotozAlgoTask):
    """Subtask to run RAIL BPZ algorithm for p(z) estimation.

    See https://github.com/LSSTDESC/rail_bpz/blob/main/src/rail/estimation/algos/bphotoz_lite.py
    for algorithm implementation.
    """

    ConfigClass = EstimatePhotozBPZAlgoConfig
    _DefaultName = "estimatePhotozBPZAlgo"


class EstimatePhotozBPZConfig(EstimatePhotozTaskConfig):
    """Config for EstimatePhotozBPZTask."""

    def setDefaults(self) -> None:
        self.photoz_algo.retarget(EstimatePhotozBPZAlgoTask)
        self.photoz_algo.stage_name = "bpz"
        self.photoz_algo.output_mode = "return"
        self.photoz_algo.bands_to_convert = ["u", "g", "r", "i", "z", "y"]
        self.photoz_algo.ref_band = self.photoz_algo.mag_template.format(band="i")
        self.photoz_algo.bands = self.photoz_algo.get_mag_name_list()
        self.photoz_algo.err_bands = self.photoz_algo.get_mag_err_name_list()
        self.photoz_algo.mag_limits = self.photoz_algo.get_mag_lim_dict()
        self.photoz_algo.filter_list = [
            "DC2LSST_u",
            "DC2LSST_g",
            "DC2LSST_r",
            "DC2LSST_i",
            "DC2LSST_z",
            "DC2LSST_y",
        ]
        self.photoz_algo.zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.photoz_algo.band_a_env = self.photoz_algo.get_band_a_env_dict()


class EstimatePhotozBPZTask(EstimatePhotozTask):
    """Task to run RAIL BPZ algorithm for p(z) estimation."""

    ConfigClass = EstimatePhotozBPZConfig
    _DefaultName = "estimatePhotozBPZ"
