# This file is part of meaz_pz
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This software is dual licensed under the GNU General Public License and also
# under a 3-clause BSD license. Recipients may choose which of these licenses
# to use; please see the files gpl-3.0.txt and/or bsd_license.txt,
# respectively.  If you choose the GPL option then the following text applies
# (but note that there is still no warranty even if you opt for BSD instead):
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

__all__ = ("ModelFormatter",)

from typing import Any

from rail.core.model import Model as RailModel
from lsst.daf.butler import FormatterV2
from lsst.resources import ResourcePath


class ModelFormatter(FormatterV2):
    """Read and write `rail.core.model.Model` objects.

    Currently assumes only local file reads are possible.
    """

    supported_write_parameters = frozenset({"format"})
    supported_extensions = frozenset({".pickle"})
    can_read_from_local_file = True

    def get_write_extension(self) -> str:
        # Default to hdf5 but allow configuration via write parameter
        format = self.write_parameters.get("format", "pickle")
        if format == "pickle":
            return ".pickle"
        # Other supported formats can be added here
        raise RuntimeError(
            f"Requested file format '{format}' is not supported for PZModel"
        )

    def read_from_local_file(
        self, path: str, component: str | None = None, expected_size: int = -1
    ) -> Any:
        return RailModel.read(path)  # type: ignore

    def write_local_file(self, in_memory_dataset: Any, uri: ResourcePath) -> None:
        in_memory_dataset.write(uri.ospath)
