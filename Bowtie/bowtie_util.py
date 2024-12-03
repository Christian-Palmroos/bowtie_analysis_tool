#!/usr/bin/env python3

import os
import numpy as np
import sys

from matplotlib import pyplot as plt

from . import bowtie

#from sixs_plot_util import *

"""

@Last updated: 2024-12-02

"""

PROTON_CHANNELS_AMOUNT = 9
PROTON_CHANNEL_START_INDEX = 8
ELECTRON_CHANNELS_AMOUNT = 7
ELECTRON_CHANNEL_START_INDEX = 1

def read_npy_vault(vault_name):
    """
    Reads in either the 'array_vault_e_256' or 'array_vault_p_256'

    Parameters:
    -----------
    vault_name : {str}

    Returns
    ----------
    particles_shot : {np.ndarray}
    particles_response : {np.ndarray}
    energy_grid : {dict}
    radiation_area : {float}
    """

    # The number of particles shot in a simulation of all energy bins
    particles_shot = np.load(f"{vault_name}/particles_Shot.npy")

    # The number of particles detected per particle channel in all energy bins
    particles_response = np.load(f"{vault_name}/particles_Respo.npy")

    other_params = np.load(f"{vault_name}/other_params.npy")

    # The total number of energy bins
    nstep = int(other_params[0])

    # The radiation area (isotropically radiating sphere) around the Geant4 instrument model in cm2
    radiation_area = other_params[2]

    # Midpoints of the energy bins in MeV
    energy_midpoint = np.load(f"{vault_name}/energy_Mid.npy")

    # High cuts of the energy bins in MeV
    energy_toppoint = np.load(f"{vault_name}/energy_Cut.npy")

    # The energy bin widths in MeV
    energy_channel_width = np.load(f"{vault_name}/energy_Width.npy")

    # An energy grid in the format compatible with the output of a function in the bowtie package
    energy_grid = { "nstep": nstep, 
                    "midpt": energy_midpoint,
                    "ehigh": energy_toppoint, 
                    "enlow": energy_toppoint - energy_channel_width,
                    "binwd": energy_channel_width }

    return particles_shot, particles_response, energy_grid, radiation_area


def assemble_response_matrix(response_df) ->list[dict]:
    """
    Assembles the response matrix needed by 'calculate_bowtie_gf()' from
    an input dataframe.
    """
    
    response_matrix = []
    for col in response_df.columns:

        response_matrix.append({
            "name": col,
            "grid": {"midpt" : response_df.index.values,
                     "nstep" : len(response_df.index)},
            "resp": response_df[col].values
        })
    
    return response_matrix


def calculate_response_matrix(particles_shot, particles_response, energy_grid:dict,
                             radiation_area:float, side:int,
                             channel_start:int, channel_stop:int,
                             contamination:bool=False, sum_channels:bool=False):
    """
    
    Parameters:
    -----------
    particles_shot : {np.ndarray}
    particles_response : {np.ndarray}
    energy_grid : {dict}
    radiation_area : {float}
    channel_start : {int}
    channel_stop : {int}
    side : {int}

    contamination : {bool} optional, default False
    sum_channels : {bool} optional, default False
    
    Returns: 
    --------
    response_matrix : {list[dict]} 
    """

    if sum_channels:
        step = 2
        channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "P1+P2", "P2", "P3+P4", "P4", "P5+P6", "P6", "P7+P8", "P8", "P9"]
    else:
        step = 1
        if contamination:
            channel_names = ["O", "EP1", "EP2", "EP3", "EP4", "EP5", "EP6", "EP7", "PE1", "PE2", "PE3", "PE4"]

        # The normal case: no summing channels and no contamination.
        else:
            channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", \
                             "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]

    response_matrix = []
    normalize_to_area = 1.0 / ((particles_shot + 1) / radiation_area) * np.pi

    for i in range(channel_start, channel_stop, step):

        if not sum_channels:
            resp_cache = particles_response[:, i, side] * normalize_to_area

        else:
            if i < channel_stop-1:
                resp_cache1 = particles_response[:, i, side] * normalize_to_area
                resp_cache2 = particles_response[:, i+1, side] * normalize_to_area

                # Sum element-wise over the slices of the two channels
                resp_cache = np.add(resp_cache1, resp_cache2)
            else:
                resp_cache = particles_response[:, i, side] * normalize_to_area

        response_matrix.append({
            "name": channel_names[i],
            "grid": energy_grid,
            "resp": resp_cache,  # The channel response
        })

    return response_matrix


def main(use_integral_bowtie = False, particle: str = 'e', side:int = 0,
         sum_channels: bool = False, contamination: bool = False,
         plot: bool = True,
         savefig: bool = False,
         save_response_matrix : bool = False,
         save_energy_and_geometry : bool = False,
         save_type : str = "csv",
         savepath : str = "geometric_factors_stats"
         ):
    """
    use_integral_bowtie : bool
    particle : str, either 'e' or 'p'
    side : int, [0,4]
    sum_channels : bool, does bowtie analysis for P1+P2, P3+P4, P5+P6, P7+P8 instead of every channel individually
    contamination : bool, tests electron/proton contamination in proton/electron channels
    plot : bool, plots the result
    savefig : bool, saves the figure
    save_response_matrix : bool, creates a file in the current directory, that contains the response matrix.
    save_type : bool, The type of file to save the effective energies and geometric facotrs to. Default 'csv'.
                        Can also be 'npy'
    """

    base_path = CURRENT_DIRECTORY # "/home/chospa/bepicolombo/bowtie-master"  # adjust to your path
    subdir = f"side{side}_response_stats"
    channels_per_decade = 256 # vault
    gamma_min = -10.0 # usually -3.5
    gamma_max = -7.0 # usually -1.5
    gamma_steps = 100
    global_emin = 0.01
    global_emax = 50.0

    # Choose which particle
    if particle == 'e':

        particle_str = "electron"
        other_particle_str = "proton"

        # Look into the response of proton channels to electron energy
        if contamination:
            instrument_channels = PROTON_CHANNELS_AMOUNT - 6 # P4, P5, P6, P7, P8 and P9 are not included here, the response is 0
            channel_start = PROTON_CHANNEL_START_INDEX

        # Normal situation: electron channel response to electron energy
        else:
            instrument_channels = ELECTRON_CHANNELS_AMOUNT
            channel_start = ELECTRON_CHANNEL_START_INDEX

    elif particle=='p':

        particle_str = "proton"
        other_particle_str = "electron"

        # Electron channel response to proton energy
        if contamination:
            instrument_channels = ELECTRON_CHANNELS_AMOUNT
            channel_start = ELECTRON_CHANNEL_START_INDEX

        # Normal situation: proton channel responses to proton energy
        else:
            instrument_channels = PROTON_CHANNELS_AMOUNT
            channel_start = PROTON_CHANNEL_START_INDEX
        

    else:
        raise ValueError("Particle needs to be 'e' or 'p'.")

    channel_stop = channel_start + instrument_channels

    # y = particles_response[:, chdraw, 2] / (particles_shot / radiation_area + 1E-24) * const.pi
    data_file_name = f"{base_path}/array_vault_{particle}_{channels_per_decade}" #.format(channels_per_decade, particle)
    print("Using response file:", data_file_name)


    particles_shot = np.load(f'{data_file_name}/particles_Shot.npy')  # The number of particles shot in a simulation of all energy bins
    particles_response = np.load(f'{data_file_name}/particles_Respo.npy')  # The number of particles detected per particle channel in all energy bins
    other_params = np.load(f'{data_file_name}/other_params.npy')
    nstep = int(other_params[0])  # The total number of energy bins
    radiation_area = other_params[2]  # The radiation area (isotropically radiating sphere) around the Geant4 instrument model in cm2
    energy_midpoint = np.load(f'{data_file_name}/energy_Mid.npy')  # midpoints of the energy bins in MeV
    energy_toppoint = np.load(f'{data_file_name}/energy_Cut.npy')  # high cuts of the energy bins in MeV
    energy_channel_width = np.load(f'{data_file_name}/energy_Width.npy')  # the energy bin widths in MeV

    # energy grid in the format compatible with the output of a function in the bowtie package
    energy_grid = { 'nstep': nstep, 'midpt': energy_midpoint,
                    'ehigh': energy_toppoint, 'enlow': energy_toppoint - energy_channel_width,
                    'binwd': energy_channel_width }

    if sum_channels:
        channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "P1+P2", "P2", "P3+P4", "P4", "P5+P6", "P6", "P7+P8", "P8", "P9"]
        step = 2
    else:
        if contamination:
            # channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "PE1", "PE2", "PE3", "PE4", "PE5", "PE6", "P7", "P8", "P9"]
            channel_names = ["O", "EP1", "EP2", "EP3", "EP4", "EP5", "EP6", "EP7", "PE1", "PE2", "PE3", "PE4"]

        # normal case: no summing channels and no contamination.
        else:
            channel_names = ["O", "E1", "E2", "E3", "E4", "E5", "E6", "E7", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9"]
        step = 1

    response_matrix = []
    normalize_to_area = 1.0 / ((particles_shot + 1) / radiation_area) * np.pi

    for i in range(channel_start, channel_stop, step):

        if not sum_channels:
            resp_cache = particles_response[:, i, side] * normalize_to_area

        else:
            if i < channel_stop-1:
                resp_cache1 = particles_response[:, i, side] * normalize_to_area
                resp_cache2 = particles_response[:, i+1, side] * normalize_to_area
                resp_cache = np.add(resp_cache1, resp_cache2) #sum element-wise over the slices of the two channels

            else:
                resp_cache = particles_response[:, i, side] * normalize_to_area

        response_matrix.append({
            "name": channel_names[i],  # last added name
            "grid": energy_grid,
            "resp": resp_cache,  # channel response
        })

    # power_law_spectra = bowtie.generate_pwlaw_spectra(energy_grid, gamma_min, gamma_max, gamma_steps)
    power_law_spectra = bowtie.generate_exppowlaw_spectra(energy_grid, gamma_min, gamma_max, gamma_steps, cutoff_energy = 0.002)
    gf_to_print = np.zeros(len(response_matrix))
    eff_energies_to_print = np.zeros(len(response_matrix))
    # gf_std = np.zeros(len(response_matrix))

    # This dictionary holds the geometric factor, its (relative) errors and the effective energy of a channel
    # gf_and_eff_en = np.zeros((len(response_matrix), 4))
    gf_and_eff_en = {}

    for channel, response in enumerate(response_matrix):
        (gf_to_print[channel], gf_std, eff_energies_to_print[channel], boundary_low, boundary_high) = bowtie.calculate_bowtie_gf(response,
                                                                                                  power_law_spectra,
                                                                                                  emin = global_emin,
                                                                                                  emax = global_emax,
                                                                                                  gamma_index_steps = gamma_steps,
                                                                                                  use_integral_bowtie = use_integral_bowtie,
                                                                                                  sigma = 3,
                                                                                                  return_gf_stddev=True,
                                                                                                  plot=False)
        print(f"Channel {response['name']}: G = {gf_to_print[channel]:.3g}, cm2srMeV; E = {eff_energies_to_print[channel]:.2g}, MeV")
        # print(f"Error limits (energy): [{boundary_low}, {boundary_high}]")
        gf_std["gfup"] -= gf_to_print[channel]
        gf_std["gflo"] -= gf_to_print[channel]
        gf_std["gflo"] = -gf_std["gflo"]
        print(f"GF_std: [{gf_std}]")
        gf_and_eff_en[response["name"]] = (eff_energies_to_print[channel], gf_to_print[channel], gf_std["gfup"], gf_std["gflo"])

    titles = (  f"Channel response as a function of {particle_str} energy",
                "Combined channel response as a function of particle energy",
                f"{other_particle_str.capitalize()} channel response as a function of {particle_str} energy"
             )

    # If we want to save the response matrix to a file, use these lists to contain
    channel_name_list = []
    incident_energies = []
    responses = []
    if plot:

        # Logic: (particle, contamination) map to correct limits in e and g
        elims_choice = {
            ('e', False) : ELECTRON_ELIMS,
            ('e', True) : P_CONTAMINATION_ELIMS,
            ('p', False) : PROTON_ELIMS,
            ('p', True) : E_CONTAMINATION_ELIMS
        }

        glims_choice = {
            ('e', False) : ELECTRON_GLIMS,
            ('e', True) : P_CONTAMINATION_GLIMS,
            ('p', False) : PROTON_GLIMS,
            ('p', True) : E_CONTAMINATION_GLIMS
        }

        fig, ax = plt.subplots(figsize=FIGSIZE)

        # elims = ELECTRON_ELIMS if particle=='e' and not contamination else PROTON_ELIMS if not contamination else CONTAMINATION_ELIMS
        ax.set_xlim(elims_choice[(particle,contamination)])

        # glims = ELECTRON_GLIMS if particle=='e' and not contamination else PROTON_GLIMS if not contamination else CONTAMINATION_GLIMS
        ax.set_ylim(glims_choice[(particle,contamination)])
        # ax.set_ylim((1e-6, 1e-1))

        title_index = 0 if not sum_channels and not contamination else 1 if sum_channels and not contamination else 2
        ax.set_title(titles[title_index], fontsize=FONTSIZES["title"])

        # Plotting the curves
        for response in response_matrix:

            if save_response_matrix:
                channel_name_list.append(response["name"])
                incident_energies.append(response["grid"]["midpt"])
                responses.append(response["resp"])

            # Check here the last channels that we do NOT want to plot if we're plotting cross-channel contamination
            if contamination and response["name"] in UNDESIRED_CROSS_CHANNELS:
                continue
            ax.plot(response["grid"]["midpt"], response["resp"], label=response["name"], lw=2.5)


        ax.legend(loc=UPPER_LEFT, bbox_to_anchor=(0.99,1.0), fontsize=FONTSIZES["legend"], frameon=True, fancybox=True)

        set_standard_response_plot_settings(ax=ax)

        if savefig:
            fig.savefig(fname=f"{CURRENT_DIRECTORY}{os.sep}{subdir}{os.sep}{titles[title_index].lower().replace(' ','_')}.png", facecolor="white", transparent=False, bbox_inches="tight")

        plt.show()


    if save_response_matrix:

        np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{subdir}{os.sep}{particle}_incident_energies.npy", arr=incident_energies)
        if not contamination:
            np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{subdir}{os.sep}{particle}_channel_names.npy", arr=channel_name_list)
            np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{subdir}{os.sep}{particle}_channel_responses.npy", arr=responses)
        else:
            np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{subdir}{os.sep}{particle}_contamination_channel_names.npy", arr=channel_name_list)
            np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{subdir}{os.sep}{other_particle_str[0]}_channel_responses_to_{particle}.npy", arr=responses)

        print(f"Succesfully created files 'channel_names.npy', 'incident_energies.npy' and channel_responses.npy' in {CURRENT_DIRECTORY}{os.sep}{subdir}!")

    if save_energy_and_geometry:

        geom_factors_dir = savepath

        if save_type == "csv":
            import pandas as pd

            if not os.path.isdir(f"{CURRENT_DIRECTORY}{os.sep}{geom_factors_dir}"):
                os.mkdir(f"{CURRENT_DIRECTORY}{os.sep}{geom_factors_dir}")
                print(f"Created {geom_factors_dir}")

            df = pd.DataFrame(data=gf_and_eff_en)
            df.index = ('E', "GF", "GF+", "GF-")

            particle = particle if not contamination and particle=='e' else "pe"

            df.to_csv(f"{CURRENT_DIRECTORY}{os.sep}{savepath}{os.sep}sixsp_side{side}_{particle}_gf_en.csv")

        elif save_type == "npy":

            if not os.path.isdir(f"{CURRENT_DIRECTORY}{os.sep}{geom_factors_dir}"):
                os.mkdir(f"{CURRENT_DIRECTORY}{os.sep}{geom_factors_dir}")
                print(f"Created {geom_factors_dir}")

            if not contamination:
                np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{geom_factors_dir}{os.sep}side{side}_{particle}_gf_en.npy", arr=gf_and_eff_en)
            else:
                np.save(file=f"{CURRENT_DIRECTORY}{os.sep}{geom_factors_dir}{os.sep}side{side}_p{particle}_gf_en.npy", arr=gf_and_eff_en)
        
        else:
            raise ValueError(f"The parameter save_type has to be either 'csv' or 'npy', not {save_type}!")

if __name__ == "__main__":

    particle = 'e'
    contamination = False

    if len(sys.argv) > 1:
        particle = 'e' if sys.argv[1] == '-e' else 'p'

    # main(particle=particle, side=side, contamination=True,
    #         sum_channels=False, plot=False, save_response_matrix=False, savefig=False,
    #         save_energy_and_geometry=True)

    for side in (0,1):
        print(f"Side {side}:")
        main(particle=particle, side=side, contamination=contamination,
            sum_channels=False, plot=False, 
            save_response_matrix=False, savefig=False,
            save_energy_and_geometry=True,
            savepath="soft_spectra_bowties")
