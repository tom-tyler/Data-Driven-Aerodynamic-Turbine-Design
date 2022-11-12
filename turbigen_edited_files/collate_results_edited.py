"""Read in optimised solutions."""

import numpy as np
import matplotlib.pyplot as plt
import os, json
import shutil


def find_all(name, path):
    """Return all files under `path` with the filename `name`."""
    results = []
    # datapoints = []
    for root, dirs, files in os.walk(path):
        if name in files:
            results.append(os.path.join(root, name))
            # datapoints.append(dirs)
    return results


def extract_from_dict(k, d):
    return np.array([di[k] for di in d])


def heatmap(
    data,
    row_labels,
    col_labels,
    ax=None,
    cbar_kw={},
    cbarlabel="",
    **kwargs
):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(
        top=True, bottom=False, labeltop=True, labelbottom=False
    )

    # Rotate the tick labels and set their alignment.
    plt.setp(
        ax.get_xticklabels(),
        rotation=-30,
        ha="right",
        rotation_mode="anchor",
    )

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", linestyle="-", linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(
    im,
    data=None,
    valfmt="{x:.2f}",
    textcolors=("black", "white"),
    threshold=None,
    **textkw
):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.0

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    # if isinstance(valfmt, str):
    #     valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(
                color=textcolors[int(im.norm(data[i, j]) > threshold)]
            )
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

job_name=str(input('Enter job name: '))

current_directory=os.getcwd()

if not os.path.exists('Results'):
    os.makedirs('Results')

job_directory = os.path.join(os.path.join(current_directory,'Results'), job_name + '_results')
if not os.path.exists(job_directory):
    os.makedirs(job_directory)

# Load all metadata files by searching for specific file name
metadata = []

for i,path in enumerate(find_all("optimised_meta.json", "run")):
    with open(path, "r") as f:
        metadata.append(json.load(f))
        runid = extract_from_dict("runid",metadata)

    datapoint_directory = os.path.dirname(path)
    
    new_directory = os.path.join(job_directory, runid[-1])
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
        
    os.replace(path, os.path.join(new_directory, "optimised_meta.json"))
    os.replace(os.path.join(datapoint_directory, 'conv.pdf'), os.path.join(new_directory, 'conv.pdf'))
    os.replace(os.path.join(datapoint_directory, 'datum.json'), os.path.join(new_directory, 'datum.json'))
    os.replace(os.path.join(datapoint_directory, 'err.pdf'), os.path.join(new_directory, 'err.pdf'))
    os.replace(os.path.join(datapoint_directory, 'log_opt.txt'), os.path.join(new_directory, 'log_opt.txt'))
    os.replace(os.path.join(datapoint_directory, 'optimised_params.json'), os.path.join(new_directory, 'optimised_params.json'))
    os.replace(os.path.join(datapoint_directory, 'sens.pdf'), os.path.join(new_directory, 'sens.pdf'))
    os.replace(os.path.join(datapoint_directory, runid[-1]), os.path.join(new_directory, runid[-1]))

print('Number of datapoints = ', len(metadata))
shutil.rmtree('run')
os.makedirs('run')

# Pull out vars of interest
phi = extract_from_dict("phi", metadata)
psi = extract_from_dict("psi", metadata)
Lam = extract_from_dict("Lam", metadata)
Co = np.mean(extract_from_dict("Co", metadata), axis=1)
Ma2 = extract_from_dict("Ma2", metadata)
eta = extract_from_dict("eta_lost_wp", metadata)
Yp_stator, Yp_rotor = extract_from_dict("Yp", metadata).T
runid = extract_from_dict("runid",metadata)

# # Make csv
if not os.path.exists('Data'):
    os.makedirs('Data')

job_file_name = job_name+'.csv'
job_data_path = os.path.join('Data',job_file_name)
M = np.column_stack((phi, psi, Lam, Ma2, Co, eta, list(map(int, runid))))
np.savetxt(job_data_path, M, delimiter=",")

fig, ax = plt.subplots()
ax.plot(Co, eta, "kx")
# ax.plot(Co, Yp_stator, "bo")
# ax.plot(Co, Yp_rotor, "kx")
plt.savefig("eta_Co.pdf")

quit()


# Histogram inputs
fig, ax = plt.subplots(2, 3)
ax[0, 0].hist(phi)
ax[0, 0].set_xlabel("$\phi$")
ax[0, 1].hist(psi)
ax[0, 1].set_xlabel("$\psi$")
ax[0, 2].hist(Lam)
ax[0, 2].set_xlabel("$\Lambda$")
ax[1, 0].hist(Ma2)
ax[1, 0].set_xlabel("$\mathit{M\kern-.25ema}_2$")
ax[1, 1].hist(Co)
ax[1, 1].set_xlabel("$C_0$")
ax[1, 2].hist(eta)
ax[1, 2].set_xlabel("$\eta$")
plt.tight_layout()
plt.savefig("plot.pdf")

# Correlations between inputs
X = np.column_stack((phi, psi, Lam, Co, Ma2, eta)).T
R = np.corrcoef(X)

fig, ax = plt.subplots()
im = ax.imshow(R, cmap="PuOr")
vnames = ("phi", "psi", "Lam", "Co", "Ma2", "eta")
heatmap(
    R,
    vnames,
    vnames,
    ax=ax,
    cmap="PuOr",
    vmin=-1,
    vmax=1,
    cbarlabel="correlation coeff.",
)
plt.savefig("cov.pdf")
