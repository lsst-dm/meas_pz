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

__all__ = [
    "EstimatePZAlgoConfigBase",
    "EstimatePZAlgoTask",
    "EstimatePZTaskConfig",
    "EstimatePZTask",
]

from abc import ABC, abstractmethod

import lsst.pex.config as pexConfig
import lsst.pipe.base.connectionTypes as cT
import numpy as np
from ceci.config import StageConfig as CeciStageConfig
from ceci.config import StageParameter as CeciParam

# from ceci.stage import PipelineStage as CeciPipelineStage
from lsst.daf.butler import DeferredDatasetHandle
from lsst.pipe.base import (
    PipelineTask,
    PipelineTaskConfig,
    PipelineTaskConnections,
    Struct,
    Task,
)
from pandas import DataFrame
from rail.core.model import Model
from rail.estimation.estimator import CatEstimator
from rail.interfaces import PZFactory


class EstimatePZConnections(
    PipelineTaskConnections, dimensions=("instrument", "tract", "patch")
):
    """Connections for tasks that make p(z) estimates

    These will take pickled model file as a "calibration-like" input,
    an objectTable as input, and create a p(z) file in 'qp' format.
    """

    pzModel = cT.PrerequisiteInput(
        doc="Model for PZ Estimation",
        name="pzModel",
        storageClass="PZModel",
        dimensions=["instrument"],
        isCalibration=True,
        # lookupFunction=_pzModelLookup,
    )

    objectTable = cT.Input(
        doc="Object table in parquet format, per patch",
        name="objectTable",
        storageClass="DataFrame",
        dimensions=(
            "skymap",
            "tract",
            "patch",
        ),
        deferLoad=True,
    )

    pzEnsemble = cT.Output(
        doc="Per-object p(z) estimates, per patch",
        name="pzEnsemble",
        storageClass="QPEnsemble",
        dimensions=(
            "skymap",
            "tract",
            "patch",
        ),
    )


class EstimatePZAlgoConfigBase(
    pexConfig.Config,
):
    """Base class for configurations of algorithm specific p(z)
    estimation tasks.

    This class mostly just translates the RAIL configuration
    parameters to pex.config parameters.

    Subclasses will just have to set
    `estimator_class` and invoke _make_fields.
    """

    @classmethod
    @abstractmethod
    def estimator_class(cls) -> type[CatEstimator]:
        raise NotImplementedError()

    stage_name = pexConfig.Field(doc="Rail stage name", dtype=str)
    mag_offset = pexConfig.Field(doc="Magnitude offset", dtype=float, default=31.4)
    flux_column_template = pexConfig.Field[str](
        doc="Template for flux column names",
        default="{band}_gaap1p0Flux",
    )
    flux_err_column_template = pexConfig.Field[str](
        doc="Template for flux error column names",
        default="{band}_gaap1p0FluxErr",
    )
    mag_template = pexConfig.Field[str](
        doc="Template for magnitude names",
        default="mag_{band}_lsst",
    )
    mag_err_template = pexConfig.Field[str](
        doc="Template for magntitude error names",
        default="mag_err_{band}_lsst",
    )
    band_a_env = pexConfig.DictField[str, float](
        doc="Reddening parameters",
        default=dict(
            u=4.81,
            g=3.64,
            r=2.70,
            i=2.06,
            z=1.58,
            y=1.31,
        ),
    )

    @classmethod
    def _make_fields(cls):
        """import the RAIL estimation stage
        and loop through the stage config parameters and make corresponding
        pex.config parameters.
        """
        stage_class = cls.estimator_class()
        for key, val in stage_class.config_options.items():
            if isinstance(val, CeciStageConfig):
                val = val.get(key)
            if isinstance(val, CeciParam):
                if val.dtype in [int, float, str]:
                    setattr(
                        cls,
                        key,
                        pexConfig.Field(
                            doc=val.msg, dtype=val.dtype, default=val.default
                        ),
                    )
                elif val.dtype in [list]:
                    setattr(
                        cls,
                        key,
                        pexConfig.ListField(
                            doc=val.msg, dtype=str, default=val.default
                        ),
                    )
                elif val.dtype in [dict]:
                    setattr(
                        cls,
                        key,
                        pexConfig.DictField(
                            doc=val.msg, keytype=str, default=val.default
                        ),
                    )


class EstimatePZAlgoTask(Task, ABC):
    """Task for algorithm specific p(z) estimation

    This will provide almost all of the functionality
    needed to run RAIL p(z) algorithms

    """

    ConfigClass = EstimatePZAlgoConfigBase

    mag_conv = np.log(10) * 0.4

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def _flux_to_mag(
        flux_vals: np.array,
        mag_offset: float,
        nondetect_val: float,
    ) -> np.array:
        """Convert flux to magnitude

        Parameters
        ----------
        flux_vals : np.array
            Input flux values (units?)

        mag_offset : float
            Magnitude offset (corresponding to a flux of 1.)

        nondetect_val : float
            Value to set for non-detections

        Returns
        -------
        mags : np.array
            Magnitude values
        """
        vals = np.where(
            np.isfinite(flux_vals),
            -2.5 * np.log10(flux_vals) + mag_offset,
            nondetect_val,
        )
        vals = np.squeeze(vals)
        return vals

    @staticmethod
    def _flux_err_to_mag_err(
        flux_vals: np.array,
        flux_err_vals: np.array,
        mag_conv: float,
        nondetect_val: float,
    ) -> np.array:
        """Config flux error to magnitude error

        Parameters
        ----------
        flux_vals : np.array
            Input flux values (units?)

        flux_err_vals : np.array
            Input flux errors (units?)

        mag_conv : float
            Magntidue to flux conversion (typically np.log(10)*0.4)

        nondetect_val : float
            Value to set for non-detections

        Returns
        -------
        mags_errs : np.array
            Magnitude errors
        """
        vals = flux_err_vals / (flux_vals * mag_conv)
        return np.squeeze(np.where(np.isfinite(vals), vals, nondetect_val))

    @staticmethod
    def _deredden_mags(
        data: dict[str, np.array],
        a_env_dict: dict[str, float],
        mag_names: dict[str, str],
        nondetect_val: float,
    ) -> dict[str, np.array]:
        """Deredden the magnitdues

        Parameters
        ----------
        data: dict[str, np.array]
            Input data

        a_env_dict: dict[str, float],
            Redenning parameters for bands

        mag_names: dict[str, str]
            Mapping from bands to magnitudes

        nondetect_val : float
            Value to set for non-detections

        Returns
        -------
        mags: dict[str, np.array]
            Udpated dict with dereddened mags
        """
        ebv = data["ebv"]
        for band_, a_env_ in a_env_dict.items():
            mag_name = mag_names[band_]
            raw_mag = data[mag_name]
            dered_mag = np.where(
                np.isfinite(raw_mag),
                raw_mag - ebv * a_env_,
                nondetect_val,
            )
            data[mag_name] = dered_mag
        return data

    def _get_flux_names(self) -> dict[str, str]:
        """Return a dict mapping band to flux column name"""
        return {
            band: self.config.flux_column_template.format(band=band)
            for band in self.config.band_a_env.keys()
        }

    def _get_flux_err_names(self) -> dict[str, str]:
        """Return a dict mapping band to flux error column name"""
        return {
            band: self.config.flux_err_column_template.format(band=band)
            for band in self.config.band_a_env.keys()
        }

    def _get_mag_names(self) -> dict[str, str]:
        """Return a dict mapping band to mag column name"""
        return {
            band: self.config.mag_template.format(band=band)
            for band in self.config.band_a_env.keys()
        }

    def _get_mag_err_names(self) -> dict[str, str]:
        """Return a dict mapping band to mag error column name"""
        return {
            band: self.config.mag_err_template.format(band=band)
            for band in self.config.band_a_env.keys()
        }

    def _get_mags_and_errs(
        self,
        fluxes: DataFrame,
        mag_offset: float,
    ) -> dict[str, np.array]:
        """Fill and return a numpy dict with mags and mag errors

        Parameters
        ----------
        fluxes : DataFrame
            Input fluxes and flux errors

        mag_offset : float
            Magnitude offset (corresponding to a flux of 1.)

        Returns
        -------
        mags: dict[str, np.array]
            Numpy dict with mags and mag errors
        """
        # get all the column names we will use
        flux_names = self._get_flux_names()
        mag_names = self._get_mag_names()
        flux_err_names = self._get_flux_err_names()
        mag_err_names = self._get_mag_err_names()
        # output dict
        mag_dict = {}
        # loop over bands, make mags and mag errors and fill dict
        for band in flux_names.keys():
            fluxVals = fluxes[flux_names[band]]
            fluxErrVals = fluxes[flux_err_names[band]]
            mag_dict[mag_names[band]] = self._flux_to_mag(
                fluxVals,
                mag_offset,
                self.config.nondetect_val,
            )
            if flux_err_names:
                mag_dict[mag_err_names[band]] = self._flux_err_to_mag_err(
                    fluxVals,
                    fluxErrVals,
                    self.mag_conv,
                    self.config.nondetect_val,
                )

        # return the dict with the mags
        return mag_dict

    def init(
        self,        
    ) -> None:
       """Initialize the Task by setting up the RAIL stage
       that will do the actually computations
       """
        # pop the pipeline task config options
        # so that we can pass the rest to RAIL
        rail_kwargs = self.config.toDict().copy()
        for key in ["saveLogOutput", "stage_name", "mag_offset", "connections"]:
            rail_kwargs.pop(key, None)
        rail_kwargs["output_mode"] = "return"

        # Build the RAIL stage
        self._stage = PZFactory.build_stage_instance(
            self.config.stage_name,
            self.config.estimator_class(),
            model_path=pzModel.data,
            input_path="dummy.in",
            **rail_kwargs,
        )

    def col_names(
        self,
    ) -> list[str]:
        """Get the list of column names to read from the input data"""
        the_col_names = (
            list(self._get_flux_names().values())
            + list(self._get_flux_err_names().values())
            + ["ebv"]
        )
        return the_col_names   
        
    def run(
        self,
        pzModel: Model,
        fluxes: DataFrame,
    ) -> Struct:
        """Run a p(z) estimation algorithm

        Parameters
        ----------
        pzModel: dict[str, Any]
            Model used by the p(z) estimation algorithm

        fluxes: DataFrame
            Fluxes used to compute the redshifts

        Returns
        -------
        pz_pdfs: qp.Ensemble
            Object with the p(z) pdfs
        """
        n_obj = len(fluxes)
        # Convert fluxes to mags
        mags = self._get_mags_and_errs(fluxes, self.config.mag_offset)
        # De-redden
        mags["ebv"] = fluxes["ebv"]
        mags = self._deredden_mags(
            mags,
            self.config.band_a_env,
            self._get_mag_names(),
            self.config.nondetect_val,
        )

        # Pass the mags to RAIL and get back the p(z) pdfs
        # as a qp.Ensemble object
        pz_pdfs = PZFactory.estimate_single_pz(self._stage, mags, n_obj)
        return Struct(pzEnsemble=pz_pdfs)


class EstimatePZTaskConfig(
    PipelineTaskConfig, pipelineConnections=EstimatePZConnections
):
    """Configuration for EstimatePZTask Pipeline task

    This just allows picking and configuring of the available algorithms

    """

    pz_algo = pexConfig.ConfigurableField(
        target=EstimatePZAlgoTask,
        doc="Algorithm specific configuration p(z) estimation task",
    )


class EstimatePZTask(PipelineTask):
    """PipelineTask for p(z) estimation

    This just makes the proper algorithm specfic Task and
    passes the input data to it.

    """

    ConfigClass = EstimatePZTaskConfig
    _DefaultName = "estimatePZ"

    def __init__(self, initInputs, **kwargs):
        super().__init__(initInputs=initInputs, **kwargs)
        self.makeSubtask("pz_algo")

    def runQuantum(self, butlerQC, inputRefs, outputRefs):
        inputs = {}
        for key, val in inputs:
            if key == 'objectTable':
                # Get only the columns that we need
                inputs[key] = bulterQC.get(
                    val,
                    parameters=dict(columns=self.pz_algo.col_names()),
                )
            else:
                inputs[key] = butlerQC.get(val)        
        outputs = self.run(**inputs)
        butlerQC.put(outputs, outputRefs)

    def run(
        self,
        pzModel: Model,
        fluxes: DataFrame,
        skip_init: bool=False,
    ) -> Struct:
        if not skip_init:
            self.pz_algo.init()
            
        ret_struct = self.pz_algo.run(pzModel, fluxes)
        return Struct(pzEnsemble=ret_struct.pzEnsemble)
