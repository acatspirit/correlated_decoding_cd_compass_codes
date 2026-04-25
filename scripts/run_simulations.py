from matplotlib.lines import Line2D
import numpy as np
from pymatching import Matching
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import Normalize
from matplotlib.ticker import ScalarFormatter
from scipy import sparse, linalg
import compass_codes as cc
import csv
import pandas as pd
import os
import collections
from datetime import datetime
import sys
import glob
from scipy.optimize import curve_fit
import circuit_gen as cc_circuit
import itertools
import stim
import json



############################################
#
# Functions for getting data (from DCC too)
#
############################################

def shots_averaging(num_shots, l, eta, err_type, in_df, CD_type, file, d=None, noise_model=None, min_total_shots=300):
    """Weighted average of chunked data."""
    if in_df is None:
        in_data = pd.read_csv(file)
        data = in_data[
            (in_data['l'] == l) &
            (in_data['eta'] == eta) &
            (in_data['error_type'] == err_type) &
            (in_data['CD_type'] == CD_type)
        ]
        if d is not None:
            data = data[data['d'] == d]
        if noise_model is not None and 'noise_model' in data.columns:
            data = data[data['noise_model'] == noise_model]
    else:
        data = in_df.copy()

    if num_shots is not None:
        data = data[data['num_shots'] == num_shots]

    data['weighted_errors'] = data['num_log_errors'] * data['num_shots']
    data_mean = (
        data.groupby('p', as_index=False)
            .agg({'num_shots': 'sum', 'weighted_errors': 'sum'})
    )
    data_mean['num_log_errors'] = data_mean['weighted_errors'] / data_mean['num_shots']
    data_mean = data_mean.drop(columns='weighted_errors')

    if min_total_shots is not None:
        data_mean = data_mean[data_mean['num_shots'] >= min_total_shots]
    return data_mean

def get_data(
    total_num_shots,
    d_list,
    l,
    p_list,
    eta,
    corr_type,
    circuit_data,
    noise_model="circuit_level",
    cd_type="SC",
    corr_decoding=False,
    pymatch_corr=False,
    data_file=None,
    append=False,
    chunk_size=5000,
    resume=True,
    fully_biased=False,
):
    """Generate logical error-rate data in chunks, with resume support.

    For each (d, p), this function checks `data_file` for previously completed
    chunks and only runs the remaining shots.
    """
    err_type = {0: "X", 1: "Z", 2: corr_type, 3: "TOTAL"}

    if circuit_data:
        columns = [
            "d", "num_shots", "p", "l", "eta", "error_type",
            "noise_model", "CD_type", "num_log_errors", "time_stamp"
        ]
    else:
        columns = [
            "d", "num_shots", "p", "l", "eta", "error_type",
            "num_log_errors", "time_stamp"
        ]

    all_rows = []

    expected_error_types = _get_expected_error_types(
        corr_type=corr_type,
        circuit_data=circuit_data,
        corr_decoding=corr_decoding,
        pymatch_corr=pymatch_corr,
    )

    existing_df = _safe_read_csv(data_file) if resume else None

    def flush_rows(rows_to_write):
        """Append rows to CSV immediately and force flush to disk."""
        if not rows_to_write:
            return

        if append and data_file is not None:
            chunk_df = pd.DataFrame(rows_to_write, columns=columns)
            file_exists = os.path.isfile(data_file)
            chunk_df.to_csv(
                data_file,
                mode="a",
                header=not file_exists,
                index=False,
            )

            with open(data_file, "a") as f:
                f.flush()
                os.fsync(f.fileno())

    for d in d_list:
        decoder = CorrelatedDecoder(eta, d, l, corr_type)

        for p in p_list:
            completed_shots = 0
            if resume:
                completed_shots = _get_completed_shots_for_point(
                    existing_df=existing_df,
                    d=d,
                    p=p,
                    l=l,
                    eta=eta,
                    expected_error_types=expected_error_types,
                    circuit_data=circuit_data,
                    noise_model=noise_model,
                    cd_type=cd_type,
                )

            if completed_shots >= total_num_shots:
                print(
                    f"Skipping d={d}, p={p}, eta={eta}, l={l} "
                    f"because {completed_shots}/{total_num_shots} shots already exist."
                )
                continue

            shots_done = completed_shots

            print(
                f"Resuming d={d}, p={p}, eta={eta}, l={l} "
                f"from {completed_shots}/{total_num_shots} shots."
            )

            while shots_done < total_num_shots:
                curr_num_shots = min(chunk_size, total_num_shots - shots_done)

                print(
                    f"Running d={d}, p={p}, eta={eta}, l={l}, "
                    f"shots {shots_done} -> {shots_done + curr_num_shots}"
                )

                if circuit_data:
                    log_errors_z_array = decoder.get_log_error_circuit_level(
                        np.array([p]),
                        "Z",
                        curr_num_shots,
                        noise_model,
                        cd_type,
                        corr_decoding,
                        pymatch_corr,
                        fully_biased=fully_biased
                    )
                    log_errors_x_array = decoder.get_log_error_circuit_level(
                        np.array([p]),
                        "X",
                        curr_num_shots,
                        noise_model,
                        cd_type,
                        corr_decoding,
                        pymatch_corr,
                        fully_biased=fully_biased
                    )

                    log_errors_z = np.sum(log_errors_z_array, axis=1)[0]
                    log_errors_x = np.sum(log_errors_x_array, axis=1)[0]
                    log_errors_total = np.sum(
                        np.logical_or(log_errors_x_array, log_errors_z_array),
                        axis=1,
                    )[0]

                    x_err_type, z_err_type, total_err_type = _get_expected_error_types(corr_type, circuit_data=circuit_data, corr_decoding=corr_decoding, pymatch_corr=pymatch_corr)

                    rows_for_chunk = [
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": x_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_x / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": z_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_z / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                        {
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": total_err_type,
                            "noise_model": noise_model,
                            "CD_type": cd_type,
                            "num_log_errors": log_errors_total / curr_num_shots,
                            "time_stamp": datetime.now(),
                        },
                    ]

                else:
                    errors = decoder.decoding_failures_correlated(p, curr_num_shots)

                    rows_for_chunk = []
                    for i in range(len(errors)):
                        rows_for_chunk.append({
                            "d": d,
                            "num_shots": curr_num_shots,
                            "p": p,
                            "l": l,
                            "eta": eta,
                            "error_type": err_type[i],
                            "num_log_errors": errors[i] / curr_num_shots,
                            "time_stamp": datetime.now(),
                        })

                all_rows.extend(rows_for_chunk)
                flush_rows(rows_for_chunk)

                shots_done += curr_num_shots

                print(
                    f"Saved d={d}, p={p}, eta={eta}, l={l}, "
                    f"chunk_shots={curr_num_shots}, total_done={shots_done}/{total_num_shots}"
                )

                # keep in-memory state updated too, so repeated points in one run
                # see the latest completed shots without rereading the file
                if existing_df is None:
                    existing_df = pd.DataFrame(rows_for_chunk, columns=columns)
                else:
                    existing_df = pd.concat(
                        [existing_df, pd.DataFrame(rows_for_chunk, columns=columns)],
                        ignore_index=True
                    )

    return pd.DataFrame(all_rows, columns=columns)

def write_data(
    total_num_shots,
    d_list,
    l,
    p_list,
    eta,
    ID,
    corr_type,
    circuit_data,
    noise_model="circuit_level",
    cd_type="SC",
    corr_decoding=False,
    pymatch_corr=False,
    chunk_size=5000,
    overwrite=False,
    resume=True,
    fully_biased=False
):
    """Write data incrementally to CSV while the job runs.

    Parameters
    ----------
    total_num_shots : int
        Total number of shots desired for each (d, p).
    chunk_size : int
        Number of shots to run before checkpointing to CSV.
    overwrite : bool
        If True, delete an existing file with the same ID before starting.
    """
    # if circuit_data:
    #     os.makedirs("circuit_data", exist_ok=True)
    #     if pymatch_corr:
    #         data_file = f"circuit_data/py_corr_{ID}.csv"
    #     else:
    #         data_file = f"circuit_data/circuit_level_{ID}.csv"
    # else:
    #     os.makedirs("corr_err_data", exist_ok=True)
    #     data_file = f"corr_err_data/code_cap_{ID}.csv"

    if circuit_data:
        os.makedirs("circuit_data", exist_ok=True)

        if pymatch_corr:
            prefix = "py_corr"
        elif corr_decoding:
            prefix = "corr"
        else:
            prefix = "circuit_level"

        d_tag = "_".join(str(d) for d in d_list)
        p_tag = "_".join(f"{float(p):.8f}" for p in p_list)

        data_file = (
            f"circuit_data/{prefix}"
            f"_l{l}_eta{eta}_cd{cd_type}_d{d_tag}_p{p_tag}.csv"
        )

    if overwrite and os.path.isfile(data_file):
        os.remove(data_file)

    data = get_data(
        total_num_shots=total_num_shots,
        d_list=d_list,
        l=l,
        p_list=p_list,
        eta=eta,
        corr_type=corr_type,
        circuit_data=circuit_data,
        noise_model=noise_model,
        cd_type=cd_type,
        corr_decoding=corr_decoding,
        pymatch_corr=pymatch_corr,
        data_file=data_file,
        append=True,
        chunk_size=chunk_size,
        resume=resume,
        fully_biased=fully_biased,
    )

    return data

def concat_csv(folder_path, circuit_data):
    """Combines all CSV files is in folder 'folder_path' and writes them to one common 
        'output_file'. The CSV files in folder_path are deleted.
        in: folder_path - the folder that stores all the csv files to be combined
            output_file - the file that the CSV files will be combined into
        out: no output. The folder_path files are deleted and the output_file has the files in folder_path added to it
    """
    data_files = glob.glob(os.path.join(folder_path, '*.csv'))
    df_list_XZ = []
    df_list_ZX = []
    df_list_CL = []
    
    for file in data_files:
        df = pd.read_csv(file)
        if not circuit_data: # the error types are X, Z, CORR_XZ, CORR_ZX, TOTAL, want to classify based on CORR_XZ and CORR_ZX
            if 'CORR_XZ' in df['error_type'].values:
                df_list_XZ.append(df)
            elif 'CORR_ZX' in df['error_type'].values:
                df_list_ZX.append(df)
        else:
            df_list_CL.append(df) # the error types are X_Mem and Z_Mem
    
    if circuit_data:
        new_data_CL = pd.concat(df_list_CL, ignore_index=True)
    else:
        new_data_XZ = pd.concat(df_list_XZ, ignore_index=True)
        new_data_ZX = pd.concat(df_list_ZX, ignore_index=True)
    
    output_file_XZ = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/xz_corr_err_data.csv'
    output_file_ZX = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/zx_corr_err_data.csv'
    output_file_CL = '/Users/ariannameinking/Documents/Brown_Research/correlated_error_biased_noise/circuit_data.csv'
    
    all_data_XZ = pd.DataFrame()
    all_data_ZX = pd.DataFrame()
    all_data_CL = pd.DataFrame()

    xz_exists = os.path.exists(output_file_XZ)
    zx_exists = os.path.exists(output_file_ZX)
    cl_exists = os.path.exists(output_file_CL)

    # Check if the output file already exists
    if not circuit_data:
        if xz_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_XZ)
            # Append the new data to the existing data
            all_data_XZ = pd.concat([existing_data, new_data_XZ], ignore_index=True)
        elif not xz_exists:
            # If the file doesn't exist, the new data is the combined data
            all_data_XZ = new_data_XZ

        if zx_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_ZX)
            # Append the new data to the existing data
            all_data_ZX = pd.concat([existing_data, new_data_ZX], ignore_index=True)
        elif not zx_exists:
            # If the file doesn't exist, the new data is the combined data
            all_data_ZX = new_data_ZX
        all_data_XZ.to_csv(output_file_XZ, index=False)
        all_data_ZX.to_csv(output_file_ZX, index=False)

    else:
        if cl_exists:
            # If it exists, load the existing data
            existing_data = pd.read_csv(output_file_CL)
            # Append the new data to the existing data
            all_data_CL = pd.concat([existing_data, new_data_CL], ignore_index=True)
        else:
            # If the file doesn't exist, the new data is the combined data
            all_data_CL = output_file_CL
        
        all_data_CL.to_csv(output_file_CL, index=False)

    
    for file in data_files:
        os.remove(file)

def load_and_concat_csvs(folder, pattern="*.csv"):
    files = glob.glob(f"{folder}/{pattern}")
    
    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder}")
    
    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    return full_df

def combine_chunked_data(df):
    df = df.copy()

    # Convert rate → total errors (temporarily)
    df["weighted_errors"] = df["num_log_errors"] * df["num_shots"]

    # Define grouping columns depending on data type
    group_cols = ["d", "p", "l", "eta", "error_type"]

    if "noise_model" in df.columns:
        group_cols += ["noise_model", "CD_type"]

    # Aggregate
    combined = (
        df.groupby(group_cols, as_index=False)
          .agg({
              "num_shots": "sum",
              "weighted_errors": "sum"
          })
    )

    # Convert back to rate
    combined["num_log_errors"] = (
        combined["weighted_errors"] / combined["num_shots"]
    )

    combined = combined.drop(columns="weighted_errors")

    return combined

def load_and_combine(folder, pattern="*.csv"):
    df = load_and_concat_csvs(folder, pattern)
    combined_df = combine_chunked_data(df)
    return combined_df

# use this now to merge csvs
def append_task_csvs_into_master(
    folder="circuit_data",
    master_file="circuit_data.csv",
    pattern="*.csv",
    delete_after_merge=False,
):
    """
    Append raw task CSV rows into the master CSV without touching existing rows.

    - Old master contents are left untouched.
    - New task rows are reordered to match the master header before appending.
    - Keeps time_stamp.
    - Does NOT combine shots.
    """

    files = glob.glob(os.path.join(folder, pattern))

    # Exclude the master file itself if it lives in the same folder
    master_abs = os.path.abspath(master_file)
    files = [f for f in files if os.path.abspath(f) != master_abs]

    if len(files) == 0:
        raise ValueError(f"No CSV files found in {folder}")

    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Skipping {f}: {e}")

    if len(df_list) == 0:
        raise ValueError("No readable task CSV files found.")

    new_df = pd.concat(df_list, ignore_index=True)

    # If master exists, match its exact column order
    if os.path.isfile(master_file):
        master_header = pd.read_csv(master_file, nrows=0).columns.tolist()

        # Add any missing columns to new_df
        for col in master_header:
            if col not in new_df.columns:
                new_df[col] = np.nan

        # Keep only the master columns, in master order
        new_df = new_df[master_header]

        # Append only new rows; do not rewrite old rows
        new_df.to_csv(master_file, mode="a", header=False, index=False)

    else:
        # If master does not exist yet, create it using a canonical order
        preferred_cols = [
            "d", "num_shots", "p", "l", "eta",
            "error_type", "num_log_errors", "time_stamp",
            "noise_model", "CD_type"
        ]
        existing_cols = [c for c in preferred_cols if c in new_df.columns]
        remaining_cols = [c for c in new_df.columns if c not in existing_cols]
        new_df = new_df[existing_cols + remaining_cols]

        new_df.to_csv(master_file, index=False)

    if delete_after_merge:
        for f in files:
            os.remove(f)

    return new_df

def _get_expected_error_types(corr_type, circuit_data, corr_decoding=False, pymatch_corr=False):
    """Return the list of error_type strings expected for one completed chunk."""
    if circuit_data:
        if pymatch_corr:
            return ["X_MEM_PY", "Z_MEM_PY", "TOTAL_MEM_PY"]
        elif corr_decoding:
            return ["X_MEM_CORR", "Z_MEM_CORR", "TOTAL_MEM_CORR"]
        else:
            return ["X_MEM", "Z_MEM", "TOTAL_MEM"]
    else:
        return ["X", "Z", corr_type, "TOTAL"]

def _safe_read_csv(csv_file):
    """Read CSV if it exists and is nonempty; otherwise return None."""
    if csv_file is None or (not os.path.isfile(csv_file)):
        return None
    try:
        df = pd.read_csv(csv_file)
        if df.empty:
            return None
        return df
    except Exception as e:
        print(f"Warning: could not read {csv_file}: {e}")
        return None

def _get_completed_shots_for_point(
    existing_df,
    d,
    p,
    l,
    eta,
    expected_error_types,
    circuit_data,
    noise_model=None,
    cd_type=None,
):
    """
    Return the number of shots already safely completed for one (d,p,l,eta,...)
    point in an existing per-task CSV.

    We sum num_shots separately for each expected error_type and take the minimum.
    That way, if the last write was partial/corrupted, we only count the shots that
    are present for all required error types.
    """
    if existing_df is None or existing_df.empty:
        return 0

    df = existing_df.copy()

    mask = (
        (df["d"] == d) &
        (np.round(df["p"], 12) == round(float(p), 12)) &
        (df["l"] == l) &
        (df["eta"] == eta)
    )

    if circuit_data:
        mask = mask & (df["noise_model"] == noise_model) & (df["CD_type"] == cd_type)

    df = df[mask]

    if df.empty:
        return 0

    completed_per_err = []
    for err in expected_error_types:
        err_df = df[df["error_type"] == err]
        if err_df.empty:
            completed_per_err.append(0)
        else:
            completed_per_err.append(int(err_df["num_shots"].sum()))

    completed_shots = min(completed_per_err) if completed_per_err else 0
    # completed_p = max(df["p"].unique()) if not df.empty else None
    return completed_shots#, completed_p

def get_data_DCC_chat(
    circuit_data,
    corr_decoding,
    noise_model,
    d_list,
    l_list,
    eta_list,
    cd_list,
    corr_list,
    total_num_shots,
    p_list=None,
    p_th_init_d=None,
    p_range=0.001,
    n_p=20,
    pymatch_corr=False,
    fully_biased=False,
    chunk_size=1000,
    overwrite=False,
    resume=True,
    shots_per_task=None,
):
    """
    Smaller-granularity SLURM array launcher.

    Circuit-level:
        one task = one (l, eta, cd_type, d, p)

    Code-cap correlated:
        one task = one (l, eta, corr_type, d, p)

    Parameters
    ----------
    total_num_shots : int
        Target total shots you eventually want per point.
    shots_per_task : int or None
        If set, each task contributes this many shots to its one point.
        This is strongly recommended for walltime-limited runs.
    """

    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    if "SLURM_ARRAY_TASK_COUNT" in os.environ:
        slurm_array_size = int(os.environ["SLURM_ARRAY_TASK_COUNT"])
    else:
        slurm_array_size = int(os.environ["SLURM_ARRAY_TASK_MAX"]) + 1

    print(f"Task ID: {task_id}")
    print(f"SLURM Array Size: {slurm_array_size}")

    if circuit_data:
        # Build p-grid first
        circuit_param_base = list(itertools.product(l_list, eta_list, cd_list, d_list))
        param_arr = []

        for l, eta, cd_type, d in circuit_param_base:
            if p_th_init_d is not None:
                if pymatch_corr:
                    err_key = "TOTAL_MEM_PY"
                elif corr_decoding:
                    err_key = "TOTAL_MEM_PY" # CHANGE THIS LATER IF MORE DATA
                else:
                    err_key = "TOTAL_MEM"

                p_th_init = p_th_init_d[(l, eta, err_key, cd_type, noise_model)]
                p_list_local = np.linspace(
                    max(p_th_init - p_range, 0.0),
                    min(p_th_init + p_range, 1.0),
                    n_p,
                )
            else:
                p_list_local = p_list

            for p in p_list_local:
                param_arr.append((l, eta, cd_type, d, float(p)))

        num_param_points = len(param_arr)

        if task_id >= num_param_points:
            raise ValueError(
                f"Task ID {task_id} exceeds number of parameter points {num_param_points}"
            )

        l, eta, cd_type, d, p = param_arr[task_id]
        corr_type = "None"

        if shots_per_task is None:
            # fallback behavior, but not recommended
            reps = max(1, slurm_array_size // num_param_points)
            num_shots = int(total_num_shots // reps)
        else:
            num_shots = int(shots_per_task)

        print("Running circuit-level point:")
        print(f"l={l}, eta={eta}, cd_type={cd_type}, d={d}, p={p}")
        print(f"shots this task = {num_shots}")

        write_data(
            total_num_shots=num_shots,
            d_list=[d],
            l=l,
            p_list=[p],
            eta=eta,
            ID=task_id,
            corr_type=corr_type,
            circuit_data=circuit_data,
            noise_model=noise_model,
            cd_type=cd_type,
            corr_decoding=corr_decoding,
            pymatch_corr=pymatch_corr,
            chunk_size=chunk_size,
            overwrite=overwrite,
            resume=resume,
            fully_biased=fully_biased,
        )

    elif corr_decoding and not circuit_data:
        codecap_param_base = list(itertools.product(l_list, eta_list, corr_list, d_list))
        param_arr = []

        for l, eta, corr_type, d in codecap_param_base:
            if p_th_init_d is not None:
                p_th_init = p_th_init_d[(l, eta, corr_type)]
                p_list_local = np.linspace(
                    max(p_th_init - 0.03, 0.0),
                    min(p_th_init + 0.03, 1.0),
                    n_p,
                )
            else:
                p_list_local = p_list

            for p in p_list_local:
                param_arr.append((l, eta, corr_type, d, float(p)))

        num_param_points = len(param_arr)

        if task_id >= num_param_points:
            raise ValueError(
                f"Task ID {task_id} exceeds number of parameter points {num_param_points}"
            )

        l, eta, corr_type, d, p = param_arr[task_id]
        cd_type = "SC"
        noise_model_local = "code_cap"

        if shots_per_task is None:
            reps = max(1, slurm_array_size // num_param_points)
            num_shots = int(total_num_shots // reps)
        else:
            num_shots = int(shots_per_task)

        print("Running code-cap point:")
        print(f"l={l}, eta={eta}, corr_type={corr_type}, d={d}, p={p}")
        print(f"shots this task = {num_shots}")

        write_data(
            total_num_shots=num_shots,
            d_list=[d],
            l=l,
            p_list=[p],
            eta=eta,
            ID=task_id,
            corr_type=corr_type,
            circuit_data=circuit_data,
            noise_model=noise_model_local,
            cd_type=cd_type,
            corr_decoding=corr_decoding,
            pymatch_corr=pymatch_corr,
            chunk_size=chunk_size,
            overwrite=overwrite,
            resume=resume,
            fully_biased=fully_biased,
        )

def get_data_DCC(
        circuit_data, 
        corr_decoding, 
        noise_model, 
        d_list, 
        l_list, 
        eta_list, 
        cd_list, 
        corr_list, 
        total_num_shots, 
        p_list=None, 
        p_th_init_d=None, 
        p_range = 0.001,
        pymatch_corr=False,
        chunk_size=5000,
        overwrite=False,
        resume=True,
    ):
    """ Function to get the data from the DCC using parallel SLURM arrays. Each array task will get data for a specific (l, eta, corr_type) or (l, eta, cd_type) combo.
        The total number of shots will be split evenly across the array tasks so that the total number of shots is reached upon averaging. 
        in: circuit_data - boolean, whether to get data from circuit or vector code cap
            corr_decoding - boolean, whether to get data from correlated decoding or not
            noise_model - string, the noise model to use for circuit data, either "code_cap", "phenom", or "circuit_level"
            d_list - list of distances to run
            l_list - list of elongations to run
            eta_list - list of noise biases to run
            cd_list - list of clifford deformations to run, either "SC", "XZZXonSqu", or "ZXXZonSqu". Not to be used if corr_decoding is True and circuit_data is False.
            corr_list - list of correlation types to run, either "CORR_XZ" or "CORR_ZX". Only to be used if corr_decoding is True and circuit_data is False.
            total_num_shots - int, the total number of shots after averaging. Each SLURM array task will run total_num_shots/reps shots.
            p_list - list of physical error rates to scan over. If None, will be set based on p_th_init_d
            p_th_init_d - dictionary with keys (l, eta, corr_type) or (l, eta, cd_type) and values the initial guess for the threshold. If None, will use a default value based on eta
            pymatch_corr - boolean, whether to use pymatching correlated decoder for circuit data
        out: no output, but will write data to a CSV file for each SLURM array task. Run concat_csv after all tasks are complete to combine the CSV files into output_file.
    """


    task_id = int(os.environ['SLURM_ARRAY_TASK_ID']) # will iter over the total slurm array size and points to where you are 
    slurm_array_size = int(os.environ['SLURM_ARRAY_TASK_MAX']) # the size of the slurm array, used to determine how many tasks to run, currently 1000

    print(f"Task ID: {task_id}")
    print(f"SLURM Array Size: {slurm_array_size}")


    if circuit_data and not (corr_decoding or pymatch_corr): # change this to get different data for circuit level plot
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, "TOTAL_MEM", cd_type, noise_model)]
            p_list = np.linspace(max(p_th_init - p_range, 0.0), min(p_th_init + p_range, 1.0), 40)
        
        write_data(total_num_shots=num_shots, 
                   d_list=d_list, 
                   l=l, 
                   p_list=p_list, 
                   eta=eta, 
                   ID=task_id, 
                   corr_type=corr_type, 
                   circuit_data=circuit_data, 
                   noise_model=noise_model, 
                   cd_type=cd_type, 
                   corr_decoding=corr_decoding, 
                   pymatch_corr=pymatch_corr,
                   chunk_size=chunk_size,
                   overwrite=overwrite,
                   resume=resume)
    if circuit_data and (pymatch_corr or corr_decoding):
        l_eta_cd_type_arr = list(itertools.product(l_list,eta_list,cd_list))
        reps = slurm_array_size//len(l_eta_cd_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_cd_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, cd_type = l_eta_cd_type_arr[ind] # get the l and eta from the task_id, pymatching corr should be doing an erasure channel this whole time, see what happens
        num_shots = int(total_num_shots//reps) # number of shots to sample
        print("l,eta,cd_type", l,eta, cd_type)
        corr_type = "None"
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, "TOTAL_MEM_PY", cd_type,noise_model)] # add the mem type somehow
            p_list = np.linspace(p_th_init - p_range, p_th_init + p_range, 18)
        write_data(total_num_shots=num_shots, 
                   d_list=d_list, 
                   l=l, 
                   p_list=p_list, 
                   eta=eta, 
                   ID=task_id, 
                   corr_type=corr_type, 
                   circuit_data=circuit_data, 
                   noise_model=noise_model, 
                   cd_type=cd_type, 
                   corr_decoding=corr_decoding, 
                   pymatch_corr=pymatch_corr,
                   chunk_size=chunk_size,
                   overwrite=overwrite,
                   resume=resume)

    if corr_decoding and not circuit_data: # change this to get different data for eta plot
        l_eta_corr_type_arr = list(itertools.product(l_list, eta_list, corr_list)) # list of tuples (l, eta, corr_type), currently 40
        reps = slurm_array_size//len(l_eta_corr_type_arr) # how many times to run file, num_shots each time
        ind = task_id%len(l_eta_corr_type_arr) # get the index of the task_id in the l_eta__corr_type_arr
        l, eta, corr_type = l_eta_corr_type_arr[ind] # get the l and eta from the task_id
        if p_th_init_d is not None:
            p_th_init = p_th_init_d[(l, eta, corr_type)]
            p_list = np.linspace(p_th_init - 0.03, p_th_init + 0.03, 40)
        num_shots = int(total_num_shots//reps) # number of shots to sample
        cd_type = "SC"
        noise_model = "code_cap"
        print("l,eta,corr_type", l,eta, corr_type)
        write_data(total_num_shots=num_shots, 
                   d_list=d_list, 
                   l=l, 
                   p_list=p_list, 
                   eta=eta, 
                   ID=task_id, 
                   corr_type=corr_type, 
                   circuit_data=circuit_data, 
                   noise_model=noise_model, 
                   cd_type=cd_type, 
                   corr_decoding=corr_decoding, 
                   pymatch_corr=pymatch_corr,
                   chunk_size=chunk_size,
                   overwrite=overwrite,
                   resume=resume)

    print("reps", reps)
    print("ind", ind)
    print("num_shots", num_shots)


def load_thresholds(filename):
    with open(filename, 'r') as f:
        raw_data = json.load(f)
    
    # Convert string keys back to tuples
    # We try to convert numeric parts back to int/float
    formatted_dict = {}
    for k, v in raw_data.items():
        parts = k.split("|")
        processed_key = []
        for p in parts:
            try:
                # Try int, then float, then keep as string
                processed_key.append(int(p) if p.isdigit() else float(p))
            except ValueError:
                processed_key.append(p)
        formatted_dict[tuple(processed_key)] = v
    return formatted_dict

if __name__ == "__main__":
    # Example usage:
    # l_list = [2,4,6] # elongation params, do 3 and 5 in another batch
    # d_list = [11,13,15,17,19] # code distances
    # eta_list = [100,1000] # noise bias
    # cd_list = ["SC", "ZXXZonSqu"] # clifford deformation types
    # corr_type = "TOTAL_MEM" # which type of correlation to use, depending on the type of decoder. Choose from ['CORR_XZ', 'CORR_ZX', 'TOTAL', 'TOTAL_MEM', 'TOTAL_PY_CORR', 'TOTAL_MEM_CORR']
    # error_type = "TOTAL_MEM" # which type of error to plot
    # # num_shots = 66666
    # corr_list = ['CORR_XZ', 'CORR_ZX']
    # corr_type_list = ['X_MEM', 'Z_MEM', 'TOTAL_MEM']  
    # noise_model = "circuit_level"
    # # py_corr = False # whether to use pymatching correlated decoder for circuit data
    # py_corr_list = [True, False] # whether to use pymatching correlated decoder for circuit data, do both in separate batches
    # circuit_data = True # whether circuit level or code cap data is desired
    # corr_decoding = False # whether to get data for correlated decoding (corrxz or corrzx), or circuit level (X/Z mem or X/Z mem py)
    # total_num_shots = 10**6
    # chunk_size=10**3
    # n_p = 20
    # p_range=0.00125
    # p_list = np.logspace(-2.5,-1.5,n_p)

    # get_data_DCC_chat(circuit_data=circuit_data,
    #                     corr_decoding=corr_decoding,
    #                     noise_model=noise_model,
    #                     d_list=d_list,
    #                     l_list=l_list,
    #                     eta_list=eta_list,
    #                     cd_list=cd_list,
    #                     corr_list=corr_list,
    #                     total_num_shots=total_num_shots,
    #                     p_list=p_list,
    #                     p_th_init_d=None,
    #                     pymatch_corr=py_corr,
    #                     fully_biased=True,
    #                     n_p = n_p,
    #                     p_range=p_range,
    #                     chunk_size=chunk_size,
    #                     resume=True,
    #                     shots_per_task=None,
    #                     )

    pass
