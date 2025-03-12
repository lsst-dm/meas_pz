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

"""Unit tests for meaz_pz"""

import pytest
from astropy.table import Table
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

from lsst.meas.pz.tests import utils


@pytest.mark.parametrize(
    "algo_name,model_file,estimator_class",
    [
        ("bpz", "models/hsc/model_inform_bpz_wrap.pickle", EstimatePZBPZTask),
        # (
        #    "cmnn",
        #    "models/hsc/model_inform_cmnn_wrap.pickle",
        #    EstimatePZCMNNTask
        # ),
        ("dnf", "models/hsc/model_inform_dnf_wrap.pickle", EstimatePZDNFTask),
        (
            "fzboost",
            "models/hsc/model_inform_fzboost_wrap.pickle",
            EstimatePZFZBoostTask,
        ),
        ("gpz", "models/hsc/model_inform_gpz_wrap.pickle", EstimatePZGPZTask),
        # (
        #    "lephare",
        #    "models/hsc/model_inform_lephare_wrap.pickle",
        #    EstimatePZLephareTask,
        # ),
        ("tpz", "models/hsc/model_inform_tpz_wrap.pickle", EstimatePZTPZTask),
    ],
)
def test_pz_task_hsc(
    hsc_dataset: Table,
    algo_name: str,
    model_file: str,
    estimator_class: type[EstimatePZTask],
) -> None:
    if estimator_class is None:
        pytest.skip(f"Missing {algo_name} in env")
    assert hsc_dataset is not None
    utils.do_pz_task(
        algo_name=algo_name,
        model_file=model_file,
        data=hsc_dataset,
        estimator_class=estimator_class,
        config_callback=utils.hsc_config_callback,
        check_callback=utils.hsc_check_callback,
    )


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
def test_pz_task_dc2(
    dc2_dataset: Table,
    algo_name: str,
    model_file: str,
    estimator_class: type[EstimatePZTask],
) -> None:
    if estimator_class is None:
        pytest.skip(f"Missing {algo_name} in env")
    assert dc2_dataset is not None
    utils.do_pz_task(
        algo_name=algo_name,
        model_file=model_file,
        data=dc2_dataset,
        estimator_class=estimator_class,
        config_callback=utils.dc2_config_callback,
        check_callback=utils.dc2_check_callback,
    )


@pytest.mark.parametrize(
    "algo_name,model_file,estimator_class",
    [
        ("bpz", "models/com_cam/model_inform_bpz_wrap.pickle", EstimatePZBPZTask),
        # (
        #     "cmnn",
        #     "models/com_cam/model_inform_cmnn_wrap.pickle",
        #     EstimatePZCMNNTask
        # ),
        ("dnf", "models/com_cam/model_inform_dnf_wrap.pickle", EstimatePZDNFTask),
        (
            "fzboost",
            "models/com_cam/model_inform_fzboost_wrap.pickle",
            EstimatePZFZBoostTask,
        ),
        ("gpz", "models/com_cam/model_inform_gpz_wrap.pickle", EstimatePZGPZTask),
        # (
        #     "lephare",
        #     "models/com_cam/model_inform_lephare_wrap.pickle",
        #     EstimatePZLephareTask,
        # ),
        ("tpz", "models/com_cam/model_inform_tpz_wrap.pickle", EstimatePZTPZTask),
    ],
)
def test_pz_task_com_cam(
    com_cam_dataset: Table,
    algo_name: str,
    model_file: str,
    estimator_class: type[EstimatePZTask],
) -> None:
    if estimator_class is None:
        pytest.skip(f"Missing {algo_name} in env")
    assert com_cam_dataset is not None
    utils.do_pz_task(
        algo_name=algo_name,
        model_file=model_file,
        data=com_cam_dataset,
        estimator_class=estimator_class,
        config_callback=utils.com_cam_config_callback,
        check_callback=utils.com_cam_check_callback,
    )
