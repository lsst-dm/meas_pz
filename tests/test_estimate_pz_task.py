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
from rail.core.model import Model as PZModel

from lsst.daf.butler import (
    Butler,
    DataCoordinate,
    DatasetType,
    DatasetRef,
    DimensionGroup,
    DimensionUniverse,
    FileDataset,
)
from lsst.meas.pz.estimate_pz_task_trainz import EstimatePZTrainZTask
from lsst.meas.pz.estimate_pz_task_knn import EstimatePZKNNTask

PIPELINES_DIR = os.path.join(os.path.dirname(__file__), "..", "pipelines")
TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
CI_HSC_GEN3_DIR = os.environ.get("CI_HSC_GEN3_DIR", None)
DAF_BUTLER_REPOSITORY_INDEX = os.environ.get("DAF_BUTLER_REPOSITORY_INDEX", None)
IS_S3DF = DAF_BUTLER_REPOSITORY_INDEX == "/sdf/group/rubin/shared/data-repos.yaml"


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

    def makeButler_ci_hsc(self, **kwargs):
        butler = Butler(
            os.path.abspath(os.path.join(CI_HSC_GEN3_DIR, "DATA")), **kwargs
        )
        return butler

    def makeButler_repo_dc2(self, **kwargs):
        butler = Butler(
            "/repo/dc2",
            collections=["2.2i/runs/test-med-1/w_2024_16/DM-43972"],
            **kwargs,
        )
        return butler

    @unittest.skipIf(not IS_S3DF, "Not at S3DF")
    def test_pz_task_trainz_s3df(self):
        butler = self.makeButler_repo_dc2()
        model_file_trainz = "model_inform_train_z_wrap.pickle"
        to_delete = []
        if not os.path.exists(model_file_trainz):
            os.system(
                f"curl -O https://portal.nersc.gov/cfs/lsst/PZ/pz_models/{model_file_trainz}"
            )
            to_delete.append(model_file_trainz)
        pz_model_trainz = PZModel.read(model_file_trainz)
        task_config_trainz = EstimatePZTrainZTask.ConfigClass()
        task_trainz = EstimatePZTrainZTask(True, config=task_config_trainz)
        dd = butler.getDeferred(
            "objectTable",
            skymap="DC2",
            tract=3829,
            patch=1,
        ).get(parameters=dict(columns=task_trainz.pz_algo.col_names()))
        out_trainz = task_trainz.run(pz_model_trainz, dd)
        out_trainz.pzEnsemble.write_to("output_trainz.hdf5")
        to_delete.append("output_trainz.hdf5")
        test_out = qp.read("output_trainz.hdf5")
        assert isinstance(test_out, qp.Ensemble)
        assert test_out.npdf == 29358
        for fdel_ in to_delete:
            os.unlink(fdel_)

    @unittest.skipIf(not IS_S3DF, "Not at S3DF")
    def test_pz_task_knn_s3df(self):
        butler = self.makeButler_repo_dc2()
        model_file_knn_lsst = "model_inform_knn_lsst_wrap.pickle"
        to_delete = []
        if not os.path.exists(model_file_knn_lsst):
            os.system(
                f"curl -O https://portal.nersc.gov/cfs/lsst/PZ/pz_models/{model_file_knn_lsst}"
            )
            to_delete.append(model_file_knn_lsst)
        pz_model_knn_lsst = PZModel.read(model_file_knn_lsst)
        task_config_knn = EstimatePZKNNTask.ConfigClass()
        task_knn = EstimatePZKNNTask(True, config=task_config_knn)
        task_knn.config.pz_algo.bands = [
            "mag_u_lsst",
            "mag_g_lsst",
            "mag_r_lsst",
            "mag_i_lsst",
            "mag_z_lsst",
            "mag_y_lsst",
        ]
        task_knn.config.pz_algo.band_a_env = dict(
            u=4.81, g=3.64, r=2.70, i=2.06, z=1.58, y=1.31
        )
        dd = butler.getDeferred(
            "objectTable",
            skymap="DC2",
            tract=3829,
            patch=1,
        ).get(
            parameters=dict(columns=task_knn.pz_algo.col_names()),
        )
        out_knn = task_knn.run(pz_model_knn_lsst, dd)
        out_knn.pzEnsemble.write_to("output_knn_lsst.hdf5")
        to_delete.append("output_knn_lsst.hdf5")
        test_out = qp.read("output_knn_lsst.hdf5")
        assert isinstance(test_out, qp.Ensemble)
        assert test_out.npdf == 29358
        for fdel_ in to_delete:
            os.unlink(fdel_)

    @unittest.skipIf(CI_HSC_GEN3_DIR is None, "CI_HSC_GEN3 not installed")
    def test_pz_tasks_ci_hsc(self):
        model_file_knn_hsc = "model_inform_knn_hsc_wrap.pickle"
        to_delete = []
        if not os.path.exists(f"tests/data/{model_file_knn_hsc}"):
            os.system(
                f"curl -o tests/data/{model_file_knn_hsc} "
                f"https://portal.nersc.gov/cfs/lsst/PZ/pz_models/{model_file_knn_hsc}"
            )
            to_delete.append(f"tests/data/{model_file_knn_hsc}")

        butler = self.makeButler_ci_hsc(writeable=True)
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
                os.path.join(TEST_DATA_DIR, "model_inform_train_z_wrap.pickle"),
                pzModel_trainz_datasetRef,
            ),
        )

        butler.ingest(
            FileDataset(
                os.path.join(TEST_DATA_DIR, "model_inform_knn_hsc_wrap.pickle"),
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
                os.path.join(TEST_DATA_DIR, "pz_pipeline_hsc.yaml"),
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

        for fdel_ in to_delete:
            os.unlink(fdel_)
