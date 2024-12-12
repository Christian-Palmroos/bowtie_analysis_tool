#!/usr/bin/env python3
"""
This file contains the Spectra class that is used to run bowtie analysis.
"""
__author__ = "Christian Palmroos"
__credits__ = ["Christian Palmroos", "Philipp Oleynik"]

from . import bowtie_util
from . import bowtie 

class Spectra:
    """
    Contains the information about what kind of spectra are considered for 
    the bowtie analysis.
    """

    def __init__(self, gamma_min:float, gamma_max:float, gamma_steps:int=100,
                 cutoff_energy:float=0.002) -> None:

        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.gamma_steps = gamma_steps

        self.cutoff_energy = cutoff_energy


    def __repr__(self) -> str:
        return f"{self.gamma_steps} spectra ranging from gamma={self.gamma_min} to gamma={self.gamma_max}."


    def set_spectral_indices(self, gamma_min, gamma_max) -> None:
        """
        Sets the limits of spectra. gamma_min < gamma_max
        """
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max


    def produce_power_law_spectra(self, response_df) -> None:
        """
        Produces a list of spectra, that are needed for bowtie analysis.
        
        Returns a list of dictionaries containing values for each spectrum.
        """

        # The incident energies are needed in a specific format, which is taken care of here.
        response_matrix = bowtie_util.assemble_response_matrix(response_df=response_df)
        grid = response_matrix[0]["grid"]

        # Generates the power law spectra
        power_law_spectra = bowtie.generate_exppowlaw_spectra(energy_grid_dict=grid, gamma_pow_min=self.gamma_min,
                                                              gamma_pow_max=self.gamma_max, num_steps=self.gamma_steps,
                                                              cutoff_energy=self.cutoff_energy)

        # Save the produced power law spectra to class attribute for easy access later.
        self.power_law_spectra = power_law_spectra
