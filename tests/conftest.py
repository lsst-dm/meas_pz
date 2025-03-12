import os

import pytest
from astropy.table import Table

try:
    TESTDATA_RAIL_DIR = os.environ["TESTDATA_RAIL_DIR"]
except KeyError:
    raise KeyError(
        "TESTDATA_RAIL must be installed and TESTDATA_RAIL_DIR set to run meas_pz unit tests"
    ) from None


@pytest.fixture(scope="session")
def hsc_dataset() -> Table:
    datapath = os.path.join(
        TESTDATA_RAIL_DIR, "data", "objectTable_hsc_9813_40_reduced.parq"
    )
    data = Table.read(datapath, format="parquet")
    return data


@pytest.fixture(scope="session")
def dc2_dataset() -> Table:
    datapath = os.path.join(
        TESTDATA_RAIL_DIR, "data", "objectTable_DC2_3829_1_reduced.parq"
    )
    data = Table.read(datapath, format="parquet")
    return data


@pytest.fixture(scope="session")
def com_cam_dataset() -> Table:
    datapath = os.path.join(
        TESTDATA_RAIL_DIR, "data", "objectTable_com_cam_5063_27_small_reduced.parq"
    )
    data = Table.read(datapath, format="parquet")
    return data
