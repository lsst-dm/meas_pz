# This file is part of meas_pz.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

__all__ = ["EstimatePZTask", "EstimatePZConfig"]

from rail.interfaces import PZFactory

import lsst.pipe.base.connectionTypes as cT
import lsst.pex.config as pexConfig
from lsst.pipe.base import PipelineTask, PipelineTaskConfig, PipelineTaskConnections


class EstimatePZConnections(PipelineTaskConnections, dimensions=("instrument", "tract")):

    camera = cT.PrerequisiteInput(
        doc="Camera instrument",
        name="camera",
        storageClass="Camera",
        dimensions=("instrument",),
        isCalibration=True,
    )
    
    pzModel = cT.PrerequisiteInput(
        doc="Model for PZ Estimation",
        name="",
        storageClass="pickle",
        dimensions=["instrument"],
        isCalibration=True,
        lookupFunction=_pzModelLookup,
    )
    
    sourceSchema = cT.InitInput(
        doc="",
        name="",
        storageClass="",
    )
    
    objectTable = cT.Input(
        doc="Object table in parquet format, per tract",
        name="objectTable_tract",
        storageClass="DataFrame",
        dimensions=("instrument", "tract",),
        deferLoad=True,
    )

    pzEnsemble = cT.Output(
        doc="Per-object p(z) estimates",
        name="pzEnsemblt_tract",
        storageClass="DataFrame",
        dimensions=("instrument", "tract",),
    )


class EstimatePZConfigBase(pexConfig.Config):
    """Config for EstimatePZ"""

    estimator_class = None
    estimator_module = None
    
    def __init_subclass__(cls):        
        stage_class = ceci.PipelineStage.get_stage(cls.estimator_class, cls.estimator_module)

        for key, val in stage_class.config_options():
            self._fields[key] = pexConfig.Field()



class EsimatePZTaskBase():

    

    def run(pzModel, objectTable):

        self.stage = PZFactory.build_cat_estimator_stage(
            self.config.stage_name,
            self.config.estimator_class,
            self.config.estimator_module,
            None,
            None,
        )

        color_table = self.get_colors()        
        pdfs = PZFactory.estimate_single_pz(stage, color_table)
        return pdfs

