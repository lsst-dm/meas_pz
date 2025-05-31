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

try:
    from rail.estimation.algos.bpz_lite import BPZliteEstimator
    from rail.estimation.estimator import CatEstimator
    has_bpz = True
except ImportError:
    has_bpz = False
    __all__ = []

if has_bpz:
    __all__ = [
        "EstimatePZBPZAlgoConfig",
        "EstimatePZBPZAlgoTask",
        "EstimatePZBPZTask",
        "EstimatePZBPZConfig",
    ]

    from .estimate_pz_task import (
        EstimatePZAlgoConfigBase,
        EstimatePZAlgoTask,
        EstimatePZTask,
        EstimatePZTaskConfig,
    )


    class EstimatePZBPZAlgoConfig(EstimatePZAlgoConfigBase):
        """Config for EstimatePZBPZAlgoTask

        This will select and configure the KNearNeighEstimator p(z)
        estimation algorithm

        """

        @classmethod
        def estimator_class(cls) -> type[CatEstimator]:
            return BPZliteEstimator


    EstimatePZBPZAlgoConfig._make_fields()


    class EstimatePZBPZAlgoTask(EstimatePZAlgoTask):
        """SubTask that runs RAIL BPZ algorithm for p(z) estimation

        See https://github.com/LSSTDESC/rail_bpz/blob/main/src/rail/estimation/algos/bpz_lite.py  # noqa
        for algorithm implementation.

        """

        ConfigClass = EstimatePZBPZAlgoConfig
        _DefaultName = "estimatePZBPZAlgo"


    class EstimatePZBPZConfig(EstimatePZTaskConfig):
        """Config for EstimatePZBPZTask

        Overrides setDefaults to use BPZ algorithm
        """

        def setDefaults(self) -> None:
            self.pz_algo.retarget(EstimatePZBPZAlgoTask)
            self.pz_algo.stage_name = "bpz"
            self.pz_algo.output_mode = "return"
            self.pz_algo.bands_to_convert = ["u", "g", "r", "i", "z", "y"]
            self.pz_algo.ref_band = self.pz_algo.mag_template.format(band='i')
            self.pz_algo.bands = self.pz_algo.get_mag_name_list()
            self.pz_algo.err_bands = self.pz_algo.get_mag_err_name_list()
            self.pz_algo.mag_limits = self.pz_algo.get_mag_lim_dict()
            self.pz_algo.filter_list = [
                "DC2LSST_u",
                "DC2LSST_g",
                "DC2LSST_r",
                "DC2LSST_i",
                "DC2LSST_z",
                "DC2LSST_y",
            ]
            self.pz_algo.zp_errors = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
            self.pz_algo.band_a_env = self.pz_algo.get_band_a_env_dict()


    class EstimatePZBPZTask(EstimatePZTask):
        """Task that runs RAIL BPZ algorithm for p(z) estimation"""

        ConfigClass = EstimatePZBPZConfig
        _DefaultName = "estimatePZBPZ"
