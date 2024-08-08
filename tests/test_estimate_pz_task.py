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
import unittest
import subprocess

import qp

from lsst.daf.butler import (
    Butler,
    DataCoordinate,
    DatasetType,
    DatasetRef,
    DimensionGroup,
    DimensionUniverse,
    FileDataset,
)

PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "pipelines")
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
CI_HSC_GEN3_DIR = os.environ.get("CI_HSC_GEN3_DIR", None)


class MeasPzTasksTestCase(unittest.TestCase):

    dim_universe = DimensionUniverse()

    objectTable_dimension_group = DimensionGroup(
        dim_universe,
        ["skymap", "tract", "patch"],
    )

    objectTable_datasetType = DatasetType(
        "objectTable",
        dimensions=objectTable_dimension_group,
        storageClass="DataFrame",
    )

    pzModel_dimension_group = DimensionGroup(
        dim_universe,
        ["instrument"],
    )

    pzModel_trainz_datasetType = DatasetType(
        "pzModel_trainz",
        dimensions=pzModel_dimension_group,
        storageClass="PZModel",
        isCalibration=True,
    )

    pzModel_knn_datasetType = DatasetType(
        "pzModel_knn",
        dimensions=pzModel_dimension_group,
        storageClass="PZModel",
        isCalibration=True,
    )

    def makeButler(self, **kwargs):
        butler = Butler(os.abspath(os.path.join(CI_HSC_GEN3_DIR, "DATA")), **kwargs)
        return butler

    @unittest.skipIf(CI_HSC_GEN3_DIR is None, "CI_HSC_GEN3 not installed")
    def test_hsc_pz_tasks(self):
        butler = self.makeButler(writeable=True)
        butler.registry.registerDatasetType(self.pzModel_trainz_datasetType)
        butler.registry.registerDatasetType(self.pzModel_knn_datasetType)
        butler.registry.registerRun("u/testing/pz_models")

        pzModel_trainz_datasetRef = DatasetRef(
            self.pzModel_trainz_datasetType,
            DataCoordinate.from_full_values(
                self.pzModel_dimension_group,
                ("HSC",),
            ),
            run="u/testing/pz_models",
        )

        pzModel_knn_datasetRef = DatasetRef(
            self.pzModel_knn_datasetType,
            DataCoordinate.from_full_values(
                self.pzModel_dimension_group,
                ("HSC",),
            ),
            run="u/testing/pz_models",
        )

        butler.ingest(
            FileDataset(
                os.path.join(TEST_DIR, "model_inform_train_z_wrap.pickle"),
                pzModel_trainz_datasetRef,
            ),
        )

        butler.ingest(
            FileDataset(
                os.path.join(TEST_DIR, "model_inform_knn_hsc_wrap.pickle"),
                pzModel_knn_datasetRef,
            ),
        )

        subprocess.run(
            [
                "pipetask",
                "run",
                "--register-dataset-types",
                "-b",
                os.path.join(CI_HSC_GEN3_DIR, "DATA"),
                "-i",
                "HSC/runs/ci_hsc,u/testing/pz_models",
                "-o",
                "u/testing/pz_rail_testing",
                "-p",
                os.path.join(TEST_DIR, "pz_pipeline_hsc.yaml"),
                "-d",
                "skymap='discrete/ci_hsc' AND tract=0 AND patch=69",
            ]
        )

        output_pz_train = butler.get(
            "pz_estimate_trainz",
            dict(skymap="discrete/ci_hsc", tract=0, patch=69),
            collections=["u/testing/pz_rail_testing"],
        )
        output_pz_knn = butler.get(
            "pz_estimate_knn",
            dict(skymap="discrete/ci_hsc", tract=0, patch=69),
            collections=["u/testing/pz_rail_testing"],
        )

        assert isinstance(output_pz_train, qp.Ensemble)
        assert isinstance(output_pz_knn, qp.Ensemble)

        assert output_pz_train.npdf == output_pz_knn.npdf
