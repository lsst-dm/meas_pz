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

__all__ = [
    "EstimatePhotozBPZAlgoConfig",
    "EstimatePhotozBPZAlgoTask",
    "EstimatePhotozBPZConfig",
    "EstimatePhotozBPZTask",
]

from rail.estimation.algos.bpz_lite import BPZliteEstimator
from rail.estimation.estimator import CatEstimator

import lsst.pex.config as pexConfig

from .estimate_photoz_task import (
    EstimatePhotozAlgoConfigBase,
    EstimatePhotozAlgoTask,
    EstimatePhotozTask,
    EstimatePhotozTaskConfig,
    photozAlgoRegistry,
)


class EstimatePhotozBPZAlgoConfig(EstimatePhotozAlgoConfigBase):
    """Config for EstimatePhotozBPZAlgoTask."""

    @classmethod
    def estimator_class(cls) -> type[CatEstimator]:
        return BPZliteEstimator

    @classmethod
    def stage_name(cls):
        return "bpz"

    def _finalize(self):
        super()._finalize()
        if not self.filter_list:
            self.filter_list = [f"DC2LSST_{band}" for band in self.bands_to_convert]
        if not self.zp_errors:
            self.zp_errors = [0.1] * len(self.filter_list)

    def setDefaults(self):
        super().setDefaults()
        self.filter_list = []
        self.zp_errors = []


EstimatePhotozBPZAlgoConfig._make_fields()


@pexConfig.registerConfigurable(EstimatePhotozBPZAlgoConfig.stage_name(), photozAlgoRegistry)
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
        super().setDefaults()
        name = EstimatePhotozBPZAlgoConfig.stage_name()
        self.connections.algo = name
        self.photoz_algo = name


class EstimatePhotozBPZTask(EstimatePhotozTask):
    """Task to run RAIL BPZ algorithm for p(z) estimation."""

    ConfigClass = EstimatePhotozBPZConfig
    _DefaultName = "estimatePhotozBPZ"
