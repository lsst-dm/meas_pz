# This file is part of meas_pz
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
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
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import os
import tempfile
import unittest
from typing import Any

from lsst.daf.butler import Butler, Config
from lsst.daf.butler.tests import DatastoreMock
from lsst.daf.butler.tests.utils import makeTestTempDir, removeTestTempDir
from lsst.pipe.base.tests.pipelineStepTester import PipelineStepTester

PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "pipelines")
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data", "extras")


class MeasPzExtraPipelineTestCase(unittest.TestCase):
    """Test the PZ pipeline plumbing for partially supported algorithms.

    This uses the `PipelineStepTester` to test
    a test pipeline define in tests/data/pz_pipeline_all_lsst.yaml

    This should include any algorithms that are wrapped in meas_pz.

    For now that is cmnn, gpz, dnf, fzboost, gpz, tpz, and lephare
    """

    def setUp(self) -> None:
        self.root = makeTestTempDir(TEST_DATA_DIR)
        self.maxDiff = None

    def tearDown(self) -> None:
        removeTestTempDir(self.root)

    def makeButler(self, **kwargs: Any) -> Butler:
        """Return new Butler instance on each call."""
        config = Config()

        # make separate temporary directory for registry of this instance
        tmpdir = tempfile.mkdtemp(dir=self.root)
        config["registry", "db"] = f"sqlite:///{tmpdir}/gen3.sqlite3"
        config = Butler.makeRepo(self.root, config)
        butler = Butler.from_config(config, **kwargs)
        DatastoreMock.apply(butler)
        return butler

    def test_extra_pz_pipeline(self) -> None:
        butler = self.makeButler(writeable=True)

        tester = PipelineStepTester(
            os.path.join(TEST_DATA_DIR, "pz_pipeline_all_lsst.yaml"),
            ["#all_pz"],
            [
                ("objectTable", {"skymap", "tract", "patch"}, "ArrowAstropy", False),
                ("pzModel_bpz", {"instrument"}, "PZModel", True),
                ("pzModel_dnf", {"instrument"}, "PZModel", True),
                ("pzModel_fzboost", {"instrument"}, "PZModel", True),
                ("pzModel_gpz", {"instrument"}, "PZModel", True),
                ("pzModel_tpz", {"instrument"}, "PZModel", True),
                ("pzModel_lephare", {"instrument"}, "PZModel", True),
                ("pzModel_cmnn", {"instrument"}, "PZModel", True),
            ],
            expected_inputs={
                "objectTable",
                "pzModel_bpz",
                "pzModel_dnf",
                "pzModel_fzboost",
                "pzModel_gpz",
                "pzModel_tpz",
                "pzModel_lephare",
                "pzModel_cmnn",
            },
            expected_outputs={
                "pz_estimate_bpz",
                "pz_estimate_dnf",
                "pz_estimate_fzboost",
                "pz_estimate_gpz",
                "pz_estimate_tpz",
                "pz_estimate_lephare",
                "pz_estimate_cmnn",
            },
        )
        tester.run(butler, self)
