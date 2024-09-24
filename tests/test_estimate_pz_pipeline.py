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

"""Unit tests for meaz_pz
"""

import os
import tempfile
import unittest
from typing import Any

from lsst.daf.butler import (
    Butler,
    Config,
)
from lsst.daf.butler.tests import DatastoreMock
from lsst.daf.butler.tests.utils import makeTestTempDir, removeTestTempDir
from lsst.pipe.base.tests.pipelineStepTester import PipelineStepTester

PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "pipelines")
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")


class MeasPzPipelineTestCase(unittest.TestCase):

    def setUp(self):
        self.root = makeTestTempDir(TEST_DATA_DIR)
        self.maxDiff = None

    def tearDown(self):
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

    def test_hsc_pz_pipeline(self):
        butler = self.makeButler(writeable=True)

        tester = PipelineStepTester(
            os.path.join(TEST_DATA_DIR, "pz_pipeline_hsc.yaml"),
            ["#all_pz"],
            [
                ("objectTable", {"skymap", "tract", "patch"}, "DataFrame", False),
                ("pzModel_trainz", {"instrument"}, "PZModel", True),
                ("pzModel_knn", {"instrument"}, "PZModel", True),
            ],
            expected_inputs={
                "objectTable",
                "pzModel_trainz",
                "pzModel_knn",
            },
            expected_outputs={
                "pz_estimate_knn",
                "pz_estimate_trainz",
            },
        )
        tester.run(butler, self)
