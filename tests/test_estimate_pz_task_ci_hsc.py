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
import subprocess
import unittest
from typing import Any

import qp
from lsst.daf.butler import (
    Butler,
    DataCoordinate,
    DatasetRef,
    DatasetType,
    DimensionGroup,
    DimensionUniverse,
    FileDataset,
)

PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "pipelines")
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
CI_HSC_GEN3_DIR = os.environ.get("CI_HSC_GEN3_DIR", None)
DAF_BUTLER_REPOSITORY_INDEX = os.environ.get("DAF_BUTLER_REPOSITORY_INDEX", None)
IS_S3DF = DAF_BUTLER_REPOSITORY_INDEX == "/sdf/group/rubin/shared/data-repos.yaml"
USER = os.environ.get("USER", "MysteriousStanger")


class MeasPzTasksTestCase(unittest.TestCase):
    """Test the PZ pipeline tasks for fully supported algorithms.

    This will run the pipeline tasks against CI_HSC_GEN3

    This should include any algorithms that are wrapped in meas_pz.

    For now that is knn and trainz.
    """

    dim_universe = DimensionUniverse()

    objectTable_dimension_group = DimensionGroup(
        dim_universe,
        ["skymap", "tract", "patch"],
    )

    objectTable_datasetType = DatasetType(
        "objectTable",
        dimensions=objectTable_dimension_group,
        storageClass="ArrowAstropy",
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

    dataset_types = [
        pzModel_trainz_datasetType,
        pzModel_knn_datasetType,
    ]

    model_files = [
        "models/hsc/model_inform_trainz_wrap.pickle",
        "models/hsc/model_inform_knn_wrap.pickle",
    ]

    def makeButler_ci_hsc(self, **kwargs: Any) -> Butler:
        assert CI_HSC_GEN3_DIR
        butler = Butler.from_config(
            os.path.abspath(os.path.join(CI_HSC_GEN3_DIR, "DATA")), **kwargs
        )
        return butler

    @unittest.skipIf(CI_HSC_GEN3_DIR is None, "CI_HSC_GEN3 not installed")
    def test_pz_tasks_ci_hsc(self) -> None:
        assert CI_HSC_GEN3_DIR

        butler = self.makeButler_ci_hsc(writeable=True)
        butler.registry.registerRun(f"u/{USER}/pz_models")

        for model_file_, dataset_type in zip(self.model_files, self.dataset_types):
            modelpath = os.path.abspath(
                os.path.expandvars(
                    os.path.join("${TESTDATA_RAIL_DIR}", model_file_),
                )
            )

            butler.registry.registerDatasetType(dataset_type)
            dataset_ref = DatasetRef(
                dataset_type,
                DataCoordinate.from_full_values(
                    self.pzModel_dimension_group,
                    ("HSC",),
                ),
                run=f"u/{USER}/pz_models",
            )
            butler.ingest(FileDataset(modelpath, dataset_ref))

        result = subprocess.run(
            [
                "pipetask",
                "run",
                "--register-dataset-types",
                "-b",
                os.path.join(CI_HSC_GEN3_DIR, "DATA"),
                "-i",
                f"HSC/runs/ci_hsc,u/{USER}/pz_models",
                "-o",
                f"u/{USER}/pz_rail_testing",
                "-p",
                os.path.join(TEST_DATA_DIR, "pz_pipeline_hsc.yaml"),
                "-d",
                "skymap='discrete/ci_hsc' AND tract=0 AND patch=69",
            ]
        )

        assert result.returncode == 0

        output_pz_train = butler.get(
            "pz_estimate_trainz",
            dict(skymap="discrete/ci_hsc", tract=0, patch=69),
            collections=[f"u/{USER}/pz_rail_testing"],
        )
        output_pz_knn = butler.get(
            "pz_estimate_knn",
            dict(skymap="discrete/ci_hsc", tract=0, patch=69),
            collections=[f"u/{USER}/pz_rail_testing"],
        )

        assert isinstance(output_pz_train, qp.Ensemble)
        assert isinstance(output_pz_knn, qp.Ensemble)

        assert output_pz_train.npdf == output_pz_knn.npdf

        # Success, go ahead and cleanup the butler
        subprocess.run(
            [
                "tests/cleanup.sh",
            ]
        )
