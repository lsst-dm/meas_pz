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

"""Test the PZ pipeline tasks for  partially supported algorithms.

This will run the pipeline tasks against a dataset
in /repo/dc2.

This should include any algorithms that are wrapped in meas_pz.

For now that is cmnn, gpz, dnf, fzboost, gpz, tpz, and lephare
"""

import os
from typing import Any

import pytest
from lsst.daf.butler import Butler
from lsst.meas.pz.estimate_pz_task import EstimatePZTask


try:
    from lsst.meas.pz.estimate_pz_task_bpz import EstimatePZBPZTask
except ImportError:
    EstimatePZBPZTask = None

try:
    from lsst.meas.pz.estimate_pz_task_cmnn import EstimatePZCMNNTask
except ImportError:
    EstimatePZCMNNTask = None

try:
    from lsst.meas.pz.estimate_pz_task_dnf import EstimatePZDNFTask
except ImportError:
    EstimatePZDNFTask = None

try:
    from lsst.meas.pz.estimate_pz_task_fzboost import EstimatePZFZBoostTask
except ImportError:
    EstimatePZFZBoostTask = None

try:
    from lsst.meas.pz.estimate_pz_task_gpz import EstimatePZGPZTask
except ImportError:
    EstimatePZGPZTask = None

try:
    from lsst.meas.pz.estimate_pz_task_lephare import EstimatePZLephareTask
except ImportError:
    EstimatePZLephareTask = None

try:
    from lsst.meas.pz.estimate_pz_task_tpz import EstimatePZTPZTask
except ImportError:
    EstimatePZTPZTask = None

from lsst.meas.pz.tests.utils import run_pz_task_s3df

TEST_DIR = os.path.abspath(os.path.dirname(__file__))
TEST_DATA_DIR = os.path.join(TEST_DIR, "data")
DAF_BUTLER_REPOSITORY_INDEX = os.environ.get("DAF_BUTLER_REPOSITORY_INDEX", None)
IS_S3DF = DAF_BUTLER_REPOSITORY_INDEX == "/sdf/group/rubin/shared/data-repos.yaml"


def makeButler_repo_dc2(**kwargs: Any) -> Butler:
    butler = Butler.from_config(
        "/repo/dc2",
        collections=["2.2i/runs/test-med-1/w_2024_16/DM-43972"],
        **kwargs,
    )
    return butler


@pytest.mark.parametrize(
    "algo_name,model_file,estimator_class",
    [
        ("bpz", "models/dc2/model_inform_bpz_wrap.pickle", EstimatePZBPZTask),
        # (
        #    "cmnn",
        #    "models/dc2/model_inform_cmnn_wrap.pickle",
        #    EstimatePZCMNNTask
        # ),
        ("dnf", "models/dc2/model_inform_dnf_wrap.pickle", EstimatePZDNFTask),
        (
            "fzboost",
            "models/dc2/model_inform_fzboost_wrap.pickle",
            EstimatePZFZBoostTask,
        ),
        ("gpz", "models/dc2/model_inform_gpz_wrap.pickle", EstimatePZGPZTask),
        # (
        #    "lephare",
        #    "models/dc2/model_inform_lephare_wrap.pickle",
        #    EstimatePZLephareTask,
        # ),
        ("tpz", "models/dc2/model_inform_tpz_wrap.pickle", EstimatePZTPZTask),
    ],
)
@pytest.mark.skipif(not IS_S3DF, reason="Not at S3DF")
def test_pz_task_s3df(
    algo_name: str,
    model_file: str,
    estimator_class: type[EstimatePZTask],
) -> None:
    if estimator_class is None:
        pytest.skip(f"Missing {algo_name} in env")
    butler = makeButler_repo_dc2()
    run_pz_task_s3df(algo_name, butler, model_file, estimator_class)
