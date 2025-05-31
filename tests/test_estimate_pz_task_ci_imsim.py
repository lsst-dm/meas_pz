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

MEAS_PZ_DIR = os.environ.get("MEAS_PZ_DIR")
PIPELINES_DIR = os.path.join(MEAS_PZ_DIR, "pipelines")
CI_IMSIM_DIR = os.environ.get("CI_IMSIM_DIR", None)
USER = os.environ.get("USER", "MysteriousStranger")

skymap = "discrete/ci_imsim/4k"
tract = 0
patch = 24


class MeasPzTasksTestCase(unittest.TestCase):
    """Test the PZ pipeline tasks for fully supported algorithms.

    This will run the pipeline tasks against CI_IMSIM

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

    pzModel_bpz_datasetType = DatasetType(
        "pzModel_bpz",
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

    pzModel_trainz_datasetType = DatasetType(
        "pzModel_trainz",
        dimensions=pzModel_dimension_group,
        storageClass="PZModel",
        isCalibration=True,
    )

    dataset_types = [
        pzModel_bpz_datasetType,
        pzModel_knn_datasetType,
        pzModel_trainz_datasetType,
    ]

    model_files = [
        "models/dc2/model_inform_bpz_wrap.pickle",
        "models/dc2/model_inform_knn_wrap.pickle",
        "models/dc2/model_inform_trainz_wrap.pickle",
    ]

    def makeButler_ci_imsim(self, **kwargs: Any) -> Butler:
        assert CI_IMSIM_DIR
        butler = Butler.from_config(
            os.path.abspath(os.path.join(CI_IMSIM_DIR, "DATA")), **kwargs
        )
        return butler

    @unittest.skipIf(CI_IMSIM_DIR is None, "CI_IMSIM not installed")
    def test_pz_tasks_ci_imsim(self) -> None:
        assert CI_IMSIM_DIR

        butler = self.makeButler_ci_imsim(writeable=True)
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
                    ("LSSTCam-imSim",),
                ),
                run=f"u/{USER}/pz_models",
            )
            butler.ingest(FileDataset(modelpath, dataset_ref))

        collection = f"u/{USER}/pz_rail_testing"

        result = subprocess.run(
            [
                "pipetask",
                "run",
                "--register-dataset-types",
                "-b",
                os.path.join(CI_IMSIM_DIR, "DATA"),
                "-i",
                f"LSSTCam-imSim/runs/ci_imsim,u/{USER}/pz_models",
                "-o",
                collection,
                "-p",
                os.path.join(PIPELINES_DIR, "photoz.yaml"),
                "-d",
                f"skymap='{skymap}' AND tract={tract} AND patch={patch}",
            ]
        )

        assert result.returncode == 0

        dataId = dict(skymap=skymap, tract=tract, patch=patch)

        npdf = None

        for model_name in ("bpz", "knn", "trainz"):
            output = butler.get(f"pz_estimate_{model_name}", **dataId, collections=collection)
            assert isinstance(output, qp.Ensemble)
            if npdf is None:
                npdf = output.npdf
            else:
                assert output.npdf == npdf

        # Success, go ahead and cleanup the butler
        subprocess.run(
            [
                "tests/cleanup_ci_imsim.sh",
            ]
        )
