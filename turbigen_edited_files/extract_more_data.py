"""Add additional data to files so that dimensional quantities can be extracted."""

import numpy as np
import matplotlib.pyplot as plt
import os, json
import shutil


def find_all(name, path):
    """Return all files under `path` with the filename `name`."""
    results = []

    for root, dirs, files in os.walk(path):
        if name in files:
            results.append(os.path.join(root, name))

    return results


def extract_from_dict(k, d):
    return np.array([di[k] for di in d])

current_directory=os.getcwd()

results_directory=os.path.join(current_directory,'Results')

for job in os.listdir(results_directory):

    job_directory = os.path.join(results_directory, job)
    job_name = job[:-8]
    
    # Load all metadata files by searching for specific file name
    metadata = []
    
    for i,path in enumerate(find_all("optimised_meta.json", job_directory)):
        with open(path, "r") as f:
            metadata.append(json.load(f))
            runid = extract_from_dict("runid",metadata)
    
    # Pull out vars of interest
    phi = extract_from_dict("phi", metadata)
    psi = extract_from_dict("psi", metadata)
    Lam = extract_from_dict("Lam", metadata)
    Co = np.mean(extract_from_dict("Co", metadata), axis=1)
    Ma2 = extract_from_dict("Ma2", metadata)
    eta = extract_from_dict("eta_lost_wp", metadata)
    Yp_stator, Yp_rotor = extract_from_dict("Yp", metadata).T
    runid = extract_from_dict("runid",metadata)
    zeta_stator, zeta_rotor = extract_from_dict("zeta", metadata).T
    s_cx_stator, s_cx_rotor = extract_from_dict("s_cx", metadata).T
    AR_stator, AR_rotor = extract_from_dict("AR", metadata).T
    loss_rat = extract_from_dict("loss_rat", metadata)
    Al1,Al2a,Al2b,Al3 = extract_from_dict("Al", metadata).T
    
    
    # # Make csv
    if not os.path.exists('Data Complete'):
        os.makedirs('Data Complete')
    
    job_file_name = job_name+'.csv'
    job_data_path = os.path.join('Data Complete',job_file_name)
    M = np.column_stack((phi, 
                         psi, 
                         Lam, 
                         Ma2, 
                         Co, 
                         eta,
                         list(map(int, runid)),
                         Yp_stator, 
                         Yp_rotor, 
                         zeta_stator,
                         zeta_rotor,
                         s_cx_stator,
                         s_cx_rotor,
                         AR_stator,
                         AR_rotor,
                         loss_rat,
                         Al1,
                         Al2a,
                         Al2b,
                         Al3))
    np.savetxt(job_data_path, M, delimiter=",")

quit()
