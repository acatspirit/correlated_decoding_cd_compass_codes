"""
The plotting functions used to get figures in paper. The plotting functions were substantially edited by ChatGPT.

Author: Arianna Meinking
Date: April 25, 2026
"""


from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.ticker import ScalarFormatter
import pandas as pd
import os
from scipy.optimize import curve_fit
import io
import requests
from run_simulations import shots_averaging





######################
#
# plotting functions
#
######################

def full_error_plot(full_df, curr_eta, curr_l, noise_model, CD_type, file, corr_decoding=False, py_corr=False, loglog=False, averaging=True, circuit_level=False, plot_by_l=False):
    """Make a threshold plot using entire CSV with weighting by number of shots for LER contribution.
        :param full_df: pandas DataFrame with unedited contents from CSV
        :param curr_eta: current noise bias to filter DataFrame
        :param curr_l: current elongation parameter to filter DataFrame
        :param noise_model: the type of simulation, either "code_cap", "phenom", or "circuit_level"
        :param CD_type: the type of clifford deformation used, from a list ["SC", "XZZXonSqu", "ZXXZonSqu"]
        :param py_corr: boolean whether pymatching correlated decoding was used, chooses from last of list ["CORR_XZ", "CORR_ZX", "X_MEM", "Z_MEM", "TOTAL_MEM", "X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"]
        :param file: the CSV file path, used for averaging shots if in_df is None
        :param loglog: boolean whether to use loglog scale for plotting
        :param averaging: boolean whether to average shots over the number of jobs
        :param circuit_level: boolean whether the data is from circuit level simulations. Alternative is vector simulation.
        :param plot_by_l: boolean whether to plot by elongation parameter l instead of error type

        :return: no return, shows a matplotlib plot
    """

    
    filtered_df = full_df[
    (full_df['l'] == curr_l) &
    (full_df['eta'] == curr_eta) &
    (full_df['noise_model'] == noise_model) &
    (full_df['CD_type'] == CD_type)
    ] # got rid num_shots

    if py_corr and not corr_decoding: 
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_PY', 'Z_MEM_PY', 'TOTAL_MEM_PY'])]
    elif corr_decoding:
        filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM_CORR', 'Z_MEM_CORR', 'TOTAL_MEM_CORR'])]
    else:
        if circuit_level:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X_MEM', 'Z_MEM', 'TOTAL_MEM'])]
        else:
            filtered_df = filtered_df[filtered_df['error_type'].isin(['X', 'Z', 'TOTAL', 'CORR_XZ', 'CORR_ZX'])]

    # Get unique error types and unique d values
    error_types = filtered_df['error_type'].unique()
    d_values = filtered_df['d'].unique()


    # Create a figure with subplots for each error type
    if len(error_types)%2 == 0:
        fig, axes = plt.subplots(len(error_types)//2, 2, figsize=(15, 5*len(error_types)//2))
    else:
        fig, axes = plt.subplots((len(error_types)//2)+1, 2, figsize=(15, 5*((len(error_types)//2)+1)))
    axes = axes.flatten()
    

    # Plot each error type in a separate subplot
    for i, error_type in enumerate(error_types):
        ax = axes[i]
        ax.tick_params(axis='both', which='major', labelsize=16)  # Change major tick label size
        ax.tick_params(axis='x', which='major', labelsize=10)
        error_type_df = filtered_df[filtered_df['error_type'] == error_type]
        prob_scale = get_prob_scale(error_type, curr_eta)
        # Plot each d value
        for d in d_values:
            d_df = error_type_df[error_type_df['d'] == d]
            if averaging:
                # to check that this is working, figure out how big this DF is
                d_df_mean = shots_averaging(
                                num_shots=None,
                                l=curr_l,
                                eta=curr_eta,
                                err_type=error_type,
                                in_df=d_df,
                                CD_type=CD_type,
                                file=file
                            )
                d_df_mean = d_df_mean.sort_values("p").reset_index(drop=True)
                d_df_mean = d_df_mean[d_df_mean["num_shots"] >= 300]
                if loglog:
                    ax.loglog(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}')
                    error_bars = 10**(-6)*np.ones(len(d_df_mean['num_log_errors'])) #error bars are wrong
                    ax.fill_between(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'] - error_bars, d_df_mean['num_log_errors'] + error_bars, alpha=0.2)
                else:
                    ax.plot(d_df_mean['p']*prob_scale, d_df_mean['num_log_errors'],  label=f'd={d}')
            else:
                ax.scatter(d_df['p']*prob_scale, d_df['num_log_errors'], s=2, label=f'd={d}')

        
        ax.set_title(f'Error Type: {error_type}', fontsize=20)
        ax.set_xlabel('p', fontsize=14)
        ax.set_ylabel('num_log_errors', fontsize=20)
        ax.legend()

    if circuit_level:
        fig.suptitle(f'Logical Error Rates for eta = {curr_eta}, l = {curr_l}, Deformation = {CD_type}')
    else:
        fig.suptitle(f'Logical Error Rates for eta = {curr_eta} and l = {curr_l}')
    plt.tight_layout()
    plt.show()

def threshold_plot(
    full_df,
    p_th0,
    p_range,
    curr_eta,
    curr_l,
    corr_type,
    CD_type,
    noise_model,
    file,
    circuit_level=False,
    py_corr=False,
    corr_decoding=False,
    loglog=False,
    averaging=True,
    show_threshold=True,
    show_fit=False,
):
    """ 
        Make a single threshold plot for one error type, using weighted averaging over raw chunk rows. Filters to show 
        only one l, eta, CD_type, and noise_model threshold plot. 
    """

    prob_scale = get_prob_scale(corr_type, curr_eta)

    # Filter the raw dataframe to the relevant physics slice and p-window
    filtered_df = full_df[
        (full_df["p"] > p_th0 - p_range) &
        (full_df["p"] < p_th0 + p_range) &
        (full_df["l"] == curr_l) &
        (full_df["eta"] == curr_eta) &
        (full_df["noise_model"] == noise_model) &
        (full_df["CD_type"] == CD_type)
    ].copy()

    if py_corr and not corr_decoding:
        filtered_df = filtered_df[
            filtered_df["error_type"].isin(["X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"])
        ]
    elif corr_decoding:
        filtered_df = filtered_df[
            filtered_df["error_type"].isin(["X_MEM_CORR", "Z_MEM_CORR", "TOTAL_MEM_CORR"])
        ]
    else:
        if circuit_level:
            filtered_df = filtered_df[
                filtered_df["error_type"].isin(["X_MEM", "Z_MEM", "TOTAL_MEM"])
            ]
        else:
            filtered_df = filtered_df[
                filtered_df["error_type"].isin(["X", "Z", "TOTAL", "CORR_XZ", "CORR_ZX"])
            ]

    # Keep only the requested plotted error type
    filtered_df = filtered_df[filtered_df["error_type"] == corr_type].copy()

    if filtered_df.empty:
        print("No data found for this threshold plot.")
        return

    d_values = np.sort(filtered_df["d"].unique())
    num_lines = len(d_values)

    cmap = colormaps["Blues_r"]
    color_values = np.linspace(0.1, 0.8, max(num_lines, 1))
    colors = [cmap(val) for val in color_values]

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot each d value
    for i, d in enumerate(d_values):
        d_df = filtered_df[filtered_df["d"] == d].copy()

        if d_df.empty:
            continue

        if averaging:
            # Temporary weighted combine over raw rows at equal p
            d_df_mean = shots_averaging(
                num_shots=None, 
                l=curr_l,
                eta=curr_eta,
                err_type=corr_type,
                in_df=d_df,
                CD_type=CD_type,
                file=file,
            )

            # Keep only the plotting window just in case
            d_df_mean = d_df_mean[
                (d_df_mean["p"] > p_th0 - p_range) &
                (d_df_mean["p"] < p_th0 + p_range)
            ].copy()

            if d_df_mean.empty:
                continue

            if loglog:
                ax.loglog(
                    d_df_mean["p"] * prob_scale,
                    d_df_mean["num_log_errors"],
                    label=f"d={d}",
                    color=colors[i],
                )
                error_bars = 10**(-6) * np.ones(len(d_df_mean["num_log_errors"]))
                ax.fill_between(
                    d_df_mean["p"] * prob_scale,
                    d_df_mean["num_log_errors"] - error_bars,
                    d_df_mean["num_log_errors"] + error_bars,
                    alpha=0.2,
                    color=colors[i],
                )
            else:
                ax.plot(
                    d_df_mean["p"] * prob_scale,
                    d_df_mean["num_log_errors"],
                    label=f"d={d}",
                    color=colors[i],
                )
        else:
            # Raw scatter of chunk rows
            ax.scatter(
                d_df["p"] * prob_scale,
                d_df["num_log_errors"],
                s=2,
                label=f"d={d}",
                color=colors[i],
            )

    # Threshold fit on weighted-combined temporary data
    popt, pcov = get_threshold(
        full_df=filtered_df,
        pth0=p_th0,
        p_range=p_range,
        l=curr_l,
        eta=curr_eta,
        error_type=corr_type,
        num_shots=None,      # important: do not filter by per-row chunk size
        CD_type=CD_type,
        noise_model=noise_model,
    )

    if isinstance(popt, int) and popt == 0:
        print("Threshold fit failed or insufficient data.")
        pth = None
        pth_error = None
    else:
        pth = popt[0]
        pth_error = np.sqrt(np.diag(pcov))[0]

    if show_threshold and pth is not None:
        # Use weighted-combined y scale for a cleaner threshold line height
        temp_df = filtered_df.copy()
        temp_df["weighted_errors"] = temp_df["num_log_errors"] * temp_df["num_shots"]
        temp_avg = (
            temp_df.groupby(["d", "p"], as_index=False)
                  .agg({"num_shots": "sum", "weighted_errors": "sum"})
        )
        temp_avg["num_log_errors"] = temp_avg["weighted_errors"] / temp_avg["num_shots"]

        ymax = temp_avg["num_log_errors"].max() if not temp_avg.empty else filtered_df["num_log_errors"].max()

        ax.vlines(
            pth*prob_scale,
            ymin=0,
            ymax=ymax,
            color="red",
            linestyles="--",
            label=f"pth = {pth:.3f} +/- {pth_error:.3f}",
        )

    if show_fit and pth is not None:
        for d in d_values:
            # Build fit curve over the weighted-averaged p grid for that d
            d_df = filtered_df[filtered_df["d"] == d].copy()

            if averaging:
                d_df_mean = shots_averaging(
                    num_shots=None,
                    l=curr_l,
                    eta=curr_eta,
                    err_type=corr_type,
                    in_df=d_df,
                    CD_type=CD_type,
                    file=file,
                )
                p_vals = np.sort(d_df_mean["p"].unique())
            else:
                p_vals = np.sort(d_df["p"].unique())

            if len(p_vals) == 0:
                continue

            y_fit = [threshold_fit((p, d), *popt) for p in p_vals]

            if loglog:
                ax.loglog(
                    np.array(p_vals) * prob_scale,
                    y_fit,
                    linestyle="--",
                    color="red",
                )
            else:
                ax.plot(
                    np.array(p_vals) * prob_scale,
                    y_fit,
                    linestyle="--",
                    color="red",
                )

    ax.set_title(f"Error Type: {corr_type}", fontsize=20)
    ax.set_xlabel("p", fontsize=14)
    ax.set_ylabel("num_log_errors", fontsize=20)
    ax.legend()

    fig.suptitle(f"Logical Error Rates for eta = {curr_eta} and l = {curr_l}")
    plt.tight_layout()
    plt.show()

def eta_threshold_plot(eta_df, cd_type, corr_type_list, noise_model):
    """
    To compare decoding types over various eta for different l, cd_type
    """
    
    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()

    cd_type = cd_type.strip()
    noise_model = noise_model.strip()

    df = eta_df[
        (eta_df['CD_type'] == cd_type) &
        (eta_df['noise_model'] == noise_model)
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(corr_type_list)

    # Colors
    cmap = colormaps['Blues_r']
    color_values = np.linspace(0.1, 0.8, len(l_values))
    l_colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(8.5 * num_cols, 4.8),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    # Store handles for shared legend
    legend_handles = []
    legend_labels = []

    

    for col_idx, error_type in enumerate(corr_type_list):
        ax = axes[col_idx]

        for l_idx, l in enumerate(l_values):
            mask = (
                (df['l'] == l) &
                (df['error_type'] == error_type)
            )
            df_filtered = df[mask].sort_values(by='eta')
            
            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            color = l_colors[l_idx]

            # Plot line
            line, = ax.plot(
                eta_vals,
                pth,
                label=f'l = {l}',
                color=color,
                marker='o'
            )

            # Shaded error
            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=color,
                alpha=0.2
            )

            # Only collect legend entries once
            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(f'l = {l}')


        parts = error_type.split("_")
        if len(parts) >= 2:
            title = rf"$\mathrm{{{parts[0]}}}_{{{parts[1]}}}$"
        else:
            title = rf"$\mathrm{{{error_type}}}$"

        ax.set_title(title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

        # Global title
    fig.suptitle(
        f"Threshold vs Bias Pymatching Correlated Decoder (Deformation: {cd_type})",
        fontsize=18,
        y=0.98
    )

    # Shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    # Manually leave vertical space for suptitle + legend
    fig.subplots_adjust(top=0.78, wspace=0.12)

    plt.show()

def eta_threshold_plot_totalmem_compare_deformations(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM"
):
    """
    Compare TOTAL_MEM threshold vs bias for multiple deformation types.
    One subplot per deformation type, all l values overlaid, shaded error bands,
    with one shared legend.
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'] == error_type)
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(cd_type_list)

    # Colors for different l values
    cmap = plt.get_cmap("Paired")

    # Each ℓ gets a pair: (light, dark)
    num_l = len(l_values)

    # Paired has 12 colors → 6 pairs
    if num_l > 6:
        raise ValueError("Paired colormap supports up to 6 ℓ values (12 colors total).")

    l_color_pairs = [
        (cmap(2*i), cmap(2*i + 1))  # (light, dark)
        for i in range(num_l)
]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(7 * num_cols, 5),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    # Pretty subplot titles
    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        ax = axes[col_idx]

        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            df_filtered = df_cd[df_cd['l'] == l].sort_values(by='eta')

            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            light_color, dark_color = l_color_pairs[l_idx]

            line, = ax.plot(
                eta_vals,
                pth,
                label=rf"$\ell = {l}$",
                color=dark_color,
                marker='o'
            )

            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=dark_color,
                alpha=0.2
            )

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        subplot_title = title_map.get(cd_type, cd_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

    mem_type = " " if error_type.startswith("TOTAL") else title_map[error_type[0]]
    fig.suptitle(
        f"Threshold vs Bias {mem_type} Memory",
        fontsize=18,
        y=0.98
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(top=0.78, wspace=0.12)

    plt.show()

def eta_threshold_plot_compare_error_types(
    eta_df,
    cd_type,
    error_type_list,
    noise_model
):
    """
    Compare threshold vs bias across four error types for one deformation type.
    Uses a 2x2 grid of subplots, with all l values overlaid and shaded error bands.
    Uses one shared legend across all subplots.
    """

    eta_df = eta_df.copy()

    if len(error_type_list) != 4:
        raise ValueError("error_type_list must contain exactly 4 error types.")

    # Be robust to either 'cd_type' or 'CD_type'
    if 'CD_type' in eta_df.columns:
        cd_col = 'CD_type'
    elif 'cd_type' in eta_df.columns:
        cd_col = 'cd_type'
    else:
        raise ValueError("DataFrame must contain either 'CD_type' or 'cd_type'.")

    eta_df[cd_col] = eta_df[cd_col].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type = cd_type.strip()
    noise_model = noise_model.strip()
    error_type_list = [et.strip() for et in error_type_list]

    df = eta_df[
        (eta_df[cd_col] == cd_type) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin(error_type_list))
    ]

    l_values = sorted(df['l'].unique())

    # Plasma colormap: one color per ell
    cmap = plt.get_cmap("plasma")
    color_values = np.linspace(0.1, 0.9, max(len(l_values), 1))
    l_colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    # Pretty titles
    title_map = {
        "CORR_XZ": r"$\mathrm{CORR}_{XZ}$",
        "CORR_ZX": r"$\mathrm{CORR}_{ZX}$",
        "TOTAL": r"$\mathrm{MWPM}$",
        "TOTAL_MEM": r"$\mathrm{TOTAL}_{MEM}$",
        "X_MEM": r"$X_{MEM}$",
        "Z_MEM": r"$Z_{MEM}$",
        "TOTAL_MEM_CORR": r"$\mathrm{TOTAL}_{MEM,CORR}$",
        "TOTAL_PY_CORR": r"$\mathrm{CORR}_{PY}$",
        "X_MEM_PY": r"$X_{MEM,PY}$",
        "Z_MEM_PY": r"$Z_{MEM,PY}$",
        "TOTAL_MEM_PY": r"$\mathrm{TOTAL}_{MEM,PY}$",
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
    }

    for idx, error_type in enumerate(error_type_list):
        ax = axes[idx]
        df_err = df[df['error_type'] == error_type]

        for l_idx, l in enumerate(l_values):
            df_filtered = df_err[df_err['l'] == l].sort_values(by='eta')

            if df_filtered.empty:
                continue

            eta_vals = df_filtered['eta'].to_numpy()
            pth = df_filtered['pth'].to_numpy()
            err = df_filtered['stderr'].to_numpy()

            color = l_colors[l_idx]

            line, = ax.plot(
                eta_vals,
                pth,
                label=rf"$\ell = {l}$",
                color=color,
                marker='o'
            )

            ax.fill_between(
                eta_vals,
                pth - err,
                pth + err,
                color=color,
                alpha=0.2
            )

            if idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        subplot_title = title_map.get(error_type, error_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.grid(True)

        # Panel labels (a), (b), (c), (d)
        panel_labels = ['(a)', '(b)', '(c)', '(d)']
        ax.text(
            0.02, 0.95,
            panel_labels[idx],
            transform=ax.transAxes,
            fontsize=14,
            fontweight='bold',
            va='top',
            ha='left'
        )

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((0, 0))
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # fig.suptitle(
    #     f"Threshold vs Bias ({title_map[cd_type]} Deformation)",
    #     fontsize=18,
    #     y=0.94
    # )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    fig.supxlabel(r"Noise Bias ($\eta$)", fontsize=12, y=0.05)
    fig.supylabel(r"Threshold $p_{th}$", fontsize=12, x=0.05)

    fig.subplots_adjust(top=0.84, wspace=0.2, hspace=0.2)

    plt.show()

def eta_threshold_plot_compare_deformations_and_decoder_2x2(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Compare threshold vs bias in a 2x2 grid:
      rows = deformation type
      cols = decoder type

    Left column: baseline decoder (MWPM)
    Right column: main decoder (correlated / PY)

    All l values are overlaid in each subplot.
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if len(cd_type_list) != 2:
        raise ValueError("cd_type_list must contain exactly 2 deformation types for the 2x2 layout.")

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)]
    else:
        raise ValueError(f"error_type must end with '{suffix_to_remove}', got {error_type}")

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ].copy()

    l_values = sorted(df['l'].unique())

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(12, 9),
        sharex=True,
        sharey=True
    )

    legend_handles = []
    legend_labels = []

    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    decoder_titles = ["MWPM", "Correlated MWPM"]

    for row_idx, cd_type in enumerate(cd_type_list):
        df_cd = df[df['CD_type'] == cd_type]

        for col_idx, decoder_type in enumerate([baseline_error_type, error_type]):
            ax = axes[row_idx, col_idx]

            for l_idx, l in enumerate(l_values):
                color = colors[l_idx]

                df_plot = df_cd[
                    (df_cd['l'] == l) &
                    (df_cd['error_type'] == decoder_type)
                ].sort_values(by='eta')

                if df_plot.empty:
                    continue

                eta_vals = df_plot['eta'].to_numpy()
                pth = df_plot['pth'].to_numpy()
                err = df_plot['stderr'].to_numpy()

                line, = ax.plot(
                    eta_vals,
                    pth,
                    label=rf"$\ell = {l}$",
                    color=color,
                    marker='o',
                    linestyle='-',
                    linewidth=1.8
                )

                ax.fill_between(
                    eta_vals,
                    pth - err,
                    pth + err,
                    color=color,
                    alpha=0.2
                )

                if row_idx == 0 and col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(rf"$\ell = {l}$")

            row_title = title_map.get(cd_type, cd_type)
            col_title = decoder_titles[col_idx]
            ax.set_title(f"{row_title}, {col_title}", fontsize=16)

            # Panel labels
            panel_labels = ['(a)', '(b)', '(c)', '(d)']
            panel_idx = row_idx * 2 + col_idx

            ax.text(
                0.02, 0.95,
                panel_labels[panel_idx],
                transform=ax.transAxes,
                fontsize=14,
                fontweight='bold',
                va='top',
                ha='left'
            )
            ax.set_xscale("log")
            ax.grid(True)

            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    # mem_type = "" if baseline_error_type.startswith("TOTAL") else title_map.get(baseline_error_type[0], baseline_error_type[0]) + " Memory"

    # fig.suptitle(
    #     f"Threshold vs Bias {mem_type}",
    #     fontsize=18,
    #     y=0.97
    # )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    fig.supxlabel("Noise Bias ($\\eta$)", fontsize=13)
    fig.supylabel("Threshold $p_{th}$", fontsize=13)

    fig.subplots_adjust(top=0.83, wspace=0.10, hspace=0.18)

    plt.show()

def eta_threshold_plot_compare_deformations_and_decoder(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Compare threshold vs bias for multiple deformation types.
    One subplot per deformation type, all l values overlaid.

    For each l:
      - solid line: error_type
      - dashed line: baseline error type with '{suffix_to_remove}' removed

    Example:
      error_type='TOTAL_MEM_CORR'  -> dashed comparison is 'TOTAL_MEM'
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)]   # removes the suffix
    else:
        raise ValueError(f"error_type must end with '{suffix_to_remove}', got {error_type}")

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ]

    l_values = sorted(df['l'].unique())
    num_cols = len(cd_type_list)

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]


    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(7 * num_cols, 5),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    # Pretty subplot titles
    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        ax = axes[col_idx]

        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            color=colors[l_idx]

            # Correlated decoder is main error type - dashed
            df_main = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == error_type)
            ].sort_values(by='eta')

            if not df_main.empty:
                eta_vals = df_main['eta'].to_numpy()
                pth = df_main['pth'].to_numpy()
                err = df_main['stderr'].to_numpy()

                line, = ax.plot(
                    eta_vals,
                    pth,
                    label=rf"$\ell = {l}$",
                    color=color,
                    marker='o',
                    linestyle='--'
                )

                ax.fill_between(
                    eta_vals,
                    pth - err,
                    pth + err,
                    color=color,
                    alpha=0.2
                )

                if col_idx == 0:
                    legend_handles.append(line)
                    legend_labels.append(rf"$\ell = {l}$")

            # MWPM baseline - solid
            df_base = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == baseline_error_type)
            ].sort_values(by='eta')

            if not df_base.empty:
                eta_vals_base = df_base['eta'].to_numpy()
                pth_base = df_base['pth'].to_numpy()
                err_base = df_base['stderr'].to_numpy()

                ax.plot(
                    eta_vals_base,
                    pth_base,
                    color=color,
                    marker='o',
                    linestyle='-',
                    linewidth=1.8
                )

                ax.fill_between(
                    eta_vals_base,
                    pth_base - err_base,
                    pth_base + err_base,
                    color=color,
                    alpha=0.10
                )

        subplot_title = title_map.get(cd_type, cd_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True)

    axes[0].set_ylabel("Threshold $p_{th}$", fontsize=12)

    mem_type = "" if baseline_error_type.startswith("TOTAL") else title_map[baseline_error_type[0]] + " Memory"
    fig.suptitle(
        f"Threshold vs Bias {mem_type}",
        fontsize=18,
        y=0.98
    )

    # Shared l legend
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )
    
    # Style legend for decoder type
    style_handles = [
        Line2D([0], [0], color="black", linestyle="--",  label="Correlated MWPM"),
        Line2D([0], [0], color="black", linestyle="-",  label="MWPM"),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=2,
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(top=0.74, wspace=0.12)

    plt.show()

def eta_delta_threshold_gap_plot_compare_deformations_and_decoder(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Compare threshold-vs-bias gap between two decoder types for multiple deformations.
    One subplot per deformation type, all l values overlaid.

    For each l, plots:
        delta_pth = pth(error_type) - pth(baseline_error_type)

    where baseline_error_type is formed by removing `suffix_to_remove`.

    Example:
        error_type='TOTAL_MEM_PY' -> baseline='TOTAL_MEM'

    The shaded band uses quadrature-combined stderr:
        sigma_delta = sqrt(stderr_main^2 + stderr_base^2)
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)] # MWPM with no correlations
    else:
        raise ValueError(
            f"error_type must end with '{suffix_to_remove}', got {error_type}"
        )

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ].copy()

    l_values = sorted(df['l'].unique())
    num_cols = len(cd_type_list)

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        1, num_cols,
        figsize=(7 * num_cols, 5),
        sharex=True,
        sharey=True
    )

    if num_cols == 1:
        axes = [axes]

    legend_handles = []
    legend_labels = []

    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        ax = axes[col_idx]
        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            color = colors[l_idx]

            df_main = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_main', 'stderr': 'stderr_main'}
            )

            df_base = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == baseline_error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_base', 'stderr': 'stderr_base'}
            )

            df_gap = pd.merge(df_main, df_base, on='eta', how='inner').sort_values('eta')

            if df_gap.empty:
                continue

            eta_vals = df_gap['eta'].to_numpy()
            delta_pth = ((df_gap['pth_main'] - df_gap['pth_base'])/ df_gap['pth_base']).to_numpy()
            delta_err = np.sqrt(
                df_gap['stderr_main'].to_numpy()**2 +
                df_gap['stderr_base'].to_numpy()**2
            )

            line, = ax.plot(
                eta_vals,
                delta_pth,
                label=rf"$\ell = {l}$",
                color=color,
                marker='o',
                linestyle='-'
            )

            ax.fill_between(
                eta_vals,
                delta_pth - delta_err,
                delta_pth + delta_err,
                color=color,
                alpha=0.2
            )

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        subplot_title = title_map.get(cd_type, cd_type)
        ax.set_title(subplot_title, fontsize=16)
        ax.set_xlabel("Noise Bias ($\\eta$)", fontsize=12)
        ax.set_xscale("log")
        ax.grid(True)

    axes[0].set_ylabel(r"$\frac{\Delta_{\mathrm{corr}}}{p^{\mathrm{MWPM}}_{\mathrm{th}}}$", fontsize=12)

    mem_type = "" if baseline_error_type.startswith("TOTAL") else title_map.get(
        baseline_error_type[0], baseline_error_type[0]
    ) + " Memory"

    fig.suptitle(
        f"Threshold Gap vs Bias {mem_type}",
        fontsize=18,
        y=0.98
    )

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    style_handles = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            label=rf"$\Delta_{{\mathrm{{corr}}}} = p_{{th}}^{{\mathrm{{PY}}}} - p_{{th}}^{{\mathrm{{MWPM}}}}$"
        ),
    ]
    fig.legend(
        handles=style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.84),
        ncol=1,
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(top=0.74, wspace=0.12)

    plt.show()

def eta_delta_threshold_gap_grid_compare_deformations_and_decoder(
    eta_df,
    cd_type_list,
    noise_model,
    error_type="TOTAL_MEM_PY",
    suffix_to_remove="_PY"
):
    """
    Make a 2x2 grid:
      columns = deformation types
      top row = Delta_corr
      bottom row = Delta_corr / p_th^MWPM
    """

    eta_df = eta_df.copy()

    eta_df['CD_type'] = eta_df['CD_type'].astype(str).str.strip()
    eta_df['noise_model'] = eta_df['noise_model'].astype(str).str.strip()
    eta_df['error_type'] = eta_df['error_type'].astype(str).str.strip()

    cd_type_list = [cd.strip() for cd in cd_type_list]
    noise_model = noise_model.strip()
    error_type = error_type.strip()

    if len(cd_type_list) != 2:
        raise ValueError("cd_type_list must contain exactly 2 deformation types.")

    if error_type.endswith(suffix_to_remove):
        baseline_error_type = error_type[:-len(suffix_to_remove)]
    else:
        raise ValueError(
            f"error_type must end with '{suffix_to_remove}', got {error_type}"
        )

    df = eta_df[
        (eta_df['CD_type'].isin(cd_type_list)) &
        (eta_df['noise_model'] == noise_model) &
        (eta_df['error_type'].isin([error_type, baseline_error_type]))
    ].copy()

    l_values = sorted(df['l'].unique())

    cmap = colormaps["plasma"]
    color_values = np.linspace(0.1, 0.8, max(len(l_values), 1))
    colors = [cmap(val) for val in color_values]

    fig, axes = plt.subplots(
        2, 2,
        figsize=(13, 8),
        sharex=True,
        sharey="row"
    )

    legend_handles = []
    legend_labels = []

    title_map = {
        "SC": "CSS",
        "ZXXZonSqu": "ZXXZ\u2610",
        "Z": "V",
        "X": "H",
    }

    for col_idx, cd_type in enumerate(cd_type_list):
        df_cd = df[df['CD_type'] == cd_type]

        for l_idx, l in enumerate(l_values):
            color = colors[l_idx]

            df_main = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_main', 'stderr': 'stderr_main'}
            )

            df_base = df_cd[
                (df_cd['l'] == l) &
                (df_cd['error_type'] == baseline_error_type)
            ][['eta', 'pth', 'stderr']].rename(
                columns={'pth': 'pth_base', 'stderr': 'stderr_base'}
            )

            df_gap = pd.merge(
                df_main,
                df_base,
                on='eta',
                how='inner'
            ).sort_values('eta')

            if df_gap.empty:
                continue

            eta_vals = df_gap['eta'].to_numpy()

            p_py = df_gap['pth_main'].to_numpy()
            p_mwpm = df_gap['pth_base'].to_numpy()

            err_py = df_gap['stderr_main'].to_numpy()
            err_mwpm = df_gap['stderr_base'].to_numpy()

            delta_corr = p_py - p_mwpm
            delta_corr_err = np.sqrt(err_py**2 + err_mwpm**2)

            frac_delta = delta_corr / p_mwpm

            frac_delta_err = np.sqrt(
                (err_py / p_mwpm)**2 +
                ((p_py * err_mwpm) / (p_mwpm**2))**2
            )

            # Top row: absolute delta
            ax_top = axes[0, col_idx]
            line, = ax_top.plot(
                eta_vals,
                delta_corr,
                color=color,
                marker='o',
                linestyle='-',
                label=rf"$\ell = {l}$"
            )
            ax_top.fill_between(
                eta_vals,
                delta_corr - delta_corr_err,
                delta_corr + delta_corr_err,
                color=color,
                alpha=0.2
            )

            # Bottom row: normalized delta
            ax_bottom = axes[1, col_idx]
            ax_bottom.plot(
                eta_vals,
                frac_delta,
                color=color,
                marker='o',
                linestyle='-'
            )
            ax_bottom.fill_between(
                eta_vals,
                frac_delta - frac_delta_err,
                frac_delta + frac_delta_err,
                color=color,
                alpha=0.2
            )

            if col_idx == 0:
                legend_handles.append(line)
                legend_labels.append(rf"$\ell = {l}$")

        axes[0, col_idx].set_title(title_map.get(cd_type, cd_type), fontsize=16)

    # Formatting all axes
    for row_idx in range(2):
        for col_idx in range(2):
            ax = axes[row_idx, col_idx]
            ax.set_xscale("log")
            ax.grid(True)

            # Scientific notation with multiplier shown at top of y-axis
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_powerlimits((0, 0))
            ax.yaxis.set_major_formatter(formatter)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0, 0].set_ylabel(r"$\Delta_{\mathrm{CORR}}$", fontsize=12)
    axes[1, 0].set_ylabel(
        r"$\Delta_{\mathrm{CORR}} / p^{\mathrm{MWPM}}_{\mathrm{th}}$",
        fontsize=12
    )

    fig.supxlabel(r"Noise Bias ($\eta$)", fontsize=13)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(l_values),
        fontsize=11,
        frameon=False
    )

    definition_handle = [
        Line2D(
            [0], [0],
            color="black",
            linestyle="-",
            label=(
                r"$\Delta_{\mathrm{CORR}} = "
                r"p_{\mathrm{th}}^{\mathrm{CORR}} - "
                r"p_{\mathrm{th}}^{\mathrm{MWPM}}$"
            )
        )
    ]

    fig.legend(
        handles=definition_handle,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=1,
        fontsize=11,
        frameon=False
    )

    fig.subplots_adjust(
        top=0.84,
        bottom=0.10,
        wspace=0.20,
        hspace=0.20
    )

    plt.show()

######################
#
# Threshold fitting 
#
######################

def threshold_fit(x, pth, nu, a,b,c):
    p,d = x
    X = (d**(1/nu))*(p-pth)
    return a + b*X + c*X**2

def get_threshold(
    full_df,
    pth0,
    p_range,
    l,
    eta,
    error_type,
    CD_type,
    noise_model="circuit_level",
    num_shots=None,
):
    """
    Return the threshold fit parameters and covariance using a temporary
    weighted average over chunked/raw rows.

    Parameters
    ----------
    full_df : pd.DataFrame
        Raw dataframe containing possibly many chunk rows per (d, p).
    pth0 : float
        Initial threshold guess.
    p_range : float
        Fit only points with p in (pth0 - p_range, pth0 + p_range).
    l, eta, error_type, CD_type : filters for the desired dataset.
    num_shots : int or None
        Optional exact-row filter. Usually leave as None for the new workflow.
    noise_model : str or None
        Optional filter if you want to restrict to one noise model.
    """
    print(
        f"Getting threshold for l = {l}, eta = {eta}, error type = {error_type}, "
        f"num_shots = {num_shots}, CD = {CD_type}"
    )

    df = full_df[
        (full_df["p"] < pth0 + p_range) &
        (full_df["p"] > pth0 - p_range) &
        (full_df["l"] == l) &
        (full_df["eta"] == eta) &
        (full_df["error_type"] == error_type) &
        (full_df["CD_type"] == CD_type)
    ].copy()

    if noise_model is not None and "noise_model" in df.columns:
        df = df[df["noise_model"] == noise_model]

    # For the new chunked/raw-row workflow, this should usually be None.
    if num_shots is not None:
        df = df[df["num_shots"] == num_shots]

    if df.empty:
        return 0, 0

    # Weighted combine over repeated rows at the same (d, p)
    df["weighted_errors"] = df["num_log_errors"] * df["num_shots"]

    df_avg = (
        df.groupby(["d", "p"], as_index=False)
          .agg({
              "num_shots": "sum",
              "weighted_errors": "sum"
          })
    )

    df_avg["num_log_errors"] = df_avg["weighted_errors"] / df_avg["num_shots"]
    df_avg = df_avg.drop(columns="weighted_errors")

    if df_avg.empty:
        return 0, 0

    # Need enough points to fit 5 parameters robustly
    if len(df_avg) < 6:
        print("Not enough averaged (d, p) points to fit threshold.")
        return 0, 0

    p_list = df_avg["p"].to_numpy().flatten()
    d_list = df_avg["d"].to_numpy().flatten()
    error_list = df_avg["num_log_errors"].to_numpy().flatten()

    popt, pcov = curve_fit(
        threshold_fit,
        (p_list, d_list),
        error_list,
        p0=[pth0, 0.5, 0, 0, 0],
    )

    return popt, pcov

def get_prob_scale(corr_type, eta):
    """ extract the amount to be scaled by given a noise bias and the type of error when plotting
    """
    prob_scale = {'X': 0.5/(1+eta), 'Z': (1+2*eta)/(2*(1+eta)), 'CORR_XZ': 1, 'CORR_ZX':1, 'TOTAL':1, 'TOTAL_MEM':1, 'X_MEM':  1, 'Z_MEM': 1, 'TOTAL_MEM_PY':1, 'X_MEM_PY':1, 'Z_MEM_PY':1,'TOTAL_MEM_CORR':1, 'X_MEM_CORR':1, 'Z_MEM_CORR':1} # TOTAL_MEM 4/3 factor of total mem is due to code_cap pauli channel scalling factor in stim, remove this?
    return prob_scale[corr_type]

def get_thresholds_full_dict(
    p_th_init_dict,
    p_range,
    output_file,
    threshold_csv,
    save_every_iteration=False,
):
    """
    Given a dictionary of threshold guesses, fit thresholds from the data file and append them
    to threshold_csv.

    This version is robust to using PY-labeled keys with CORR-labeled data:
        X_MEM_PY     <-> X_MEM_CORR
        Z_MEM_PY     <-> Z_MEM_CORR
        TOTAL_MEM_PY <-> TOTAL_MEM_CORR

    Behavior:
    - If no matching data exists, prints:
        "didn't get data for l={l}, eta={eta}, ..."
    - If matching data exists but threshold fit fails, prints:
        "can't get threshold for l={l}, eta={eta}, ..."
    """

    def candidate_error_types(err_type):
        """
        Return a list of candidate error_type labels to try in the data.
        First entry is the preferred label to fit with.
        """
        mapping = {
            "X_MEM_PY": ["X_MEM_PY", "X_MEM_CORR"],
            "Z_MEM_PY": ["Z_MEM_PY", "Z_MEM_CORR"],
            "TOTAL_MEM_PY": ["TOTAL_MEM_PY", "TOTAL_MEM_CORR"],
            "X_MEM_CORR": ["X_MEM_CORR", "X_MEM_PY"],
            "Z_MEM_CORR": ["Z_MEM_CORR", "Z_MEM_PY"],
            "TOTAL_MEM_CORR": ["TOTAL_MEM_CORR", "TOTAL_MEM_PY"],
        }
        return mapping.get(err_type, [err_type])

    def get_matching_subset(df, l, eta, CD_type, noise_model, err_type_candidates):
        """
        Return the subset of df matching l, eta, CD_type, noise_model, and any candidate error type.
        """
        subset = df[
            (df["l"] == l) &
            (df["eta"] == eta) &
            (df["CD_type"] == CD_type)
        ].copy()

        if "noise_model" in subset.columns and noise_model is not None:
            subset = subset[subset["noise_model"] == noise_model]

        subset = subset[subset["error_type"].isin(err_type_candidates)]
        return subset

    def choose_error_type_present(subset, err_type_candidates):
        """
        Pick the first candidate error type that is actually present in the subset.
        """
        present = set(subset["error_type"].unique())
        for err_type in err_type_candidates:
            if err_type in present:
                return err_type
        return None

    # Read files once, not inside the loop.
    df = pd.read_csv(output_file)
    all_thresholds_df = pd.read_csv(threshold_csv)

    new_rows = []

    for key, p_th_init in p_th_init_dict.items():
        l, eta, corr_type, CD_type, noise_model = key
        print(f"Trying l={l}, eta={eta}, error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}")

        err_type_candidates = candidate_error_types(corr_type)

        subset = get_matching_subset(
            df=df,
            l=l,
            eta=eta,
            CD_type=CD_type,
            noise_model=noise_model,
            err_type_candidates=err_type_candidates,
        )

        if subset.empty:
            print(
                f"didn't get data for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}"
            )
            continue

        fit_error_type = choose_error_type_present(subset, err_type_candidates)

        if fit_error_type is None:
            print(
                f"didn't get data for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}"
            )
            continue

        try:
            pop, pcov = get_threshold(
                df,
                p_th_init,
                p_range,
                l,
                eta,
                fit_error_type,
                CD_type,
            )
        except Exception as e:
            print(
                f"can't get threshold for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}. "
                f"Error: {e}"
            )
            continue

        if isinstance(pop, int) or pop is None or pcov is None:
            print(
                f"can't get threshold for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}"
            )
            continue

        try:
            threshold = pop[0]
            std_error = np.sqrt(np.diag(pcov))[0]
        except Exception as e:
            print(
                f"can't get threshold for l={l}, eta={eta}, "
                f"error_type={corr_type}, CD_type={CD_type}, noise_model={noise_model}. "
                f"Bad fit output: {e}"
            )
            continue

        print(f"Initial guess: {p_th_init}, fitted threshold: {threshold}")

        new_rows.append({
            "l": l,
            "eta": eta,
            "error_type": corr_type,   # preserve original dict key label
            "CD_type": CD_type,
            "noise_model": noise_model,
            "pth": threshold,
            "stderr": std_error,
        })

        if save_every_iteration:
            temp_df = pd.concat([all_thresholds_df, pd.DataFrame(new_rows)], ignore_index=True)
            temp_df.to_csv(threshold_csv, index=False)

    if new_rows:
        all_thresholds_df = pd.concat([all_thresholds_df, pd.DataFrame(new_rows)], ignore_index=True)

    all_thresholds_df.to_csv(threshold_csv, index=False)
    return all_thresholds_df


###############################
#
# Data loading functions
#
###############################

def get_paper_data(file_url):
    """
    Fetches a CSV from the Duke Repository DOI link and returns a DataFrame.
    
    Args:
        file_url (str): The direct download link for the CSV.
    """
    try:
        # We use a header to avoid being blocked by basic bot filters
        response = requests.get(file_url, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status() # Check for download errors
        
        # Convert the byte stream into a format pandas understands
        df = pd.read_csv(io.BytesIO(response.content))
        print(f"Successfully loaded data from {file_url}")
        return df
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def load_data_smart(filename, url):
    if os.path.exists(filename):
        print(f"Loading local file: {filename}")
        return pd.read_csv(filename)
    else:
        print("Local file not found. Fetching from Duke Repository...")
        df = get_paper_data(url)
        if df is not None:
            df.to_csv(filename, index=False) # Save it locally for next time
        return df


if __name__ == "__main__":
    # --- How to use it in your plotting script ---

    # Example: Replace with your actual direct download link from Duke
    # DUKE_CSV_URL = "https://research.repository.duke.edu/api/access/datafile/:persistentId?persistentId=doi:10.12345/DUKE/ABCDE"

    # df = load_data_smart("circuit_data.csv", DUKE_CSV_URL)

    # eta = 0.5
    # l = 3

    # curr_num_shots = chunk_size # the file has 20408 for the 3,5 and 30303 for the 2,4,6 and 52631 for pycorr
    # noise_model = "circuit_level"
    # CD_type = "SC"
    # py_corr = False # whether to use pymatching correlated decoder for circuit data
    # corr_decoding = False # whether to get data for correlated decoding using my decoder
    # error_type = "TOTAL_MEM" # which type of error to plot, choose from ['X_MEM', 'Z_MEM', 'TOTAL_MEM', 'TOTAL_PY_MEM', 'TOTAL_MEM_PY_CORR']
    # p_range = 0.00125

    # full_error_plot(
    #     full_df=df,
    #     curr_eta=eta,
    #     curr_l=l,
    #     curr_num_shots=None,
    #     noise_model=noise_model,
    #     CD_type=CD_type,
    #     file=output_file,
    #     corr_decoding=corr_decoding,
    #     py_corr=py_corr,
    #     circuit_level=circuit_data,
    #     loglog=True,
    # )

    pass

