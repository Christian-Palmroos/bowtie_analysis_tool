#!/usr/bin/env python3

import pandas as pd

# from matplotlib import pyplot as plt

from . import bowtie_util as btutil
from . import bowtie

from .Spectra import Spectra

class Bowtie:
    """
    Contains the functionality needed to run bowtie analysis.
    """

    def __init__(self, energy_min:float, energy_max:float, data:pd.DataFrame,
                 integral_bowtie:bool=False, sigma:int=3) -> None:

        if isinstance(data,pd.DataFrame):
            self.data = data
            self.response_matrix = btutil.assemble_response_matrix(response_df=data)
        else:
            raise TypeError(f"Data needs to be a pandas dataframe, {type(data)} was passed!")

        self.energy_min = energy_min
        self.energy_max = energy_max
        self.integral_bowtie = integral_bowtie
        self.sigma = sigma

    def set_energy_range(self, energy_min:float, energy_max:float) -> None:
        """
        Sets the limits of the energy range.
        """
        self.energy_min = energy_min
        self.energy_max = energy_max

    def bowtie_analysis(self, channel:str, spectra:Spectra, plot:bool=False,
                        geom_factor_confidence:float=0.9) -> dict:
        """
        Runs bowtie analysis to a channel.
        """

        # Check that the channel is valid
        if channel not in self.data.columns:
            raise ValueError(f"Channel {channel} not found in response dataframe!")

        # Choose the correct response from the matrix
        for response in self.response_matrix:
            if response["name"]==channel:
                response_dict = response

        # Check that the input spectra is valid (it has a power law spectra)
        if not isinstance(spectra,Spectra): 
            raise TypeError(f"Input spectra needs to be a Spectra-type of object, not {type(spectra)}!")
        if not hasattr(spectra, "power_law_spectra"):
            raise AttributeError("Produce power law spectra with Spectra.produce_power_law_spectra() to calculate bowtie!")

        # The bowtie_results are in order:
        # Geometric factor (G \Delta E) in cm2srMeV : {float}
        # Geometric factor errors : {dict} with keys ["gfup", "gflo"]
        # The effective energy of the channel in MeV : {float}
        # The channel effective lower boundary in MeV : {float} 
        # The channel effective upper boundary in MeV : {float}
        # if plot, also returns fig and axes
        bowtie_results = bowtie.calculate_bowtie_gf(response_data=response_dict, model_spectra=spectra.power_law_spectra,
                                                    emin=self.energy_min, emax=self.energy_max,
                                                    gamma_index_steps=spectra.gamma_steps, use_integral_bowtie=self.integral_bowtie,
                                                    sigma=self.sigma, plot=plot, gfactor_confidence_level=geom_factor_confidence,
                                                    return_gf_stddev=True, channel=channel)

        # Collect the results to a dictionary for easier handling
        result_dict = {}
        result_names = ["geometric_factor", "geometric_factor_errors", "effective_energy",\
                        "effective_lower_boundary", "effective_upper_boundary"]
        if plot:
            result_names.append("fig")
            result_names.append("axes")

        for i, res in enumerate(bowtie_results):

            # Handle errors here
            if i==1:
                res["gfup"] -= bowtie_results[0]
                res["gflo"] -= bowtie_results[0]
                res["gflo"] = -res["gflo"]

            result_dict[result_names[i]] = res

        return result_dict

    def bowtie_analysis_full_stack(self, spectra, plot:bool=False,
                                   geom_factor_confidence:float=0.9) -> list[dict]:
        """
        Wrapper for bowtie_analysis(). Runs bowtie_analysis() for all of the 
        given channels in bowtie.data
        """

        all_bowtie_results = []

        for channel in self.data.columns:

            new_result = self.bowtie_analysis(channel=channel, spectra=spectra, plot=plot,
                                              geom_factor_confidence=geom_factor_confidence)
        
            all_bowtie_results.append(new_result)

        return all_bowtie_results



def main(filename):

    response_df = pd.read_csv(filename, index_col="incident_energy")

if __name__ == "__main__":

    filename = "sixs_side0_electron_responses.csv"
    main(filename=filename)
