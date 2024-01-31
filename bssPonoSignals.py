# %%
import math
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.signal import sosfiltfilt, butter
from sklearn.decomposition import FastICA
import typing
from typing import Optional

from intanutil_rhd.read_data import read_data


# %% 
# IMPORT DATA

filepath = '5min_post_stim8_trial2_240110_153011.rhd'
data = read_data(filepath)
SAMPLE_RATE = 30000

# extract the raw 30kHz voltage (𝛍V) data
raw_30k_data = data['amplifier_data']

print(f'\nShape of data: {raw_30k_data.shape}')

# %% Helper functions

def make_samples(data, sample_rate, window_length_sec, overlap_sec):
    # Calculate the number of points in each window and the overlap
    window_length_points = int(sample_rate * window_length_sec)
    overlap_points = int(sample_rate * overlap_sec)

    # Calculate the number of samples
    num_samples = (len(data) - overlap_points) // (window_length_points - overlap_points)

    # Create an empty array to hold the samples
    samples = np.empty((num_samples, window_length_points))

    # Fill the array with overlapping samples
    for i in range(num_samples):
        start = i * (window_length_points - overlap_points)
        end = start + window_length_points
        samples[i, :] = data[start:end]

    return samples # (samples, window_length_points)

# function to build a filter given a filter order, freq range, and sampling rate
def build_filter(filt_order, filt_range, fs):
    btype = 'bandpass'
    if filt_range[0] is None:
        btype = 'lowpass'
        filt_range = filt_range[1]
    elif filt_range[1] is None:
        btype = 'highpass'
        filt_range = filt_range[0]
    
    filter = butter(filt_order, filt_range, btype=btype, analog=False, output='sos', fs=fs)
    return btype, filter
# %% Visualize raw data
# examine the amplifier data for the entire recording
def plot_30k_data(data, channels=None):
    """ function to plot the raw amplifier data.
    """
    if channels is None:
        channels = np.arange(data.shape[0])
    data = data[channels].T  

    b_type, filter = build_filter(4, (250, 5000), 30000)      # build a filter for each frequency range
    filt_data = sosfiltfilt(filter, data, axis=0)

    filt_data = filt_data[:60000]  
    data = data[:60000]  

    time = np.arange(0, data.shape[0] / 30, 1/30)

    fig, axs = plt.subplots(len(channels), 1, figsize=(15, 10))                            # figure size is (width, height), scale height to the number of channels
    for i in range(len(channels)):
        axs[i].plot(time, data[:,i])            
        # axs[i].plot(time, filt_data[:,i])            
        axs[i].set_ylabel(f'Channel {channels[i]}')
    plt.xlabel('Time (ms)')
    plt.savefig(f'raw_data.png', dpi=300, facecolor='w', transparent=False, bbox_inches='tight')
    
plot_30k_data(raw_30k_data, channels=[0,1, 2])            # plot the first 3 channels
# %% 
# Separation function

def separate(data: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]: 

    # Apply FastICA
    ica = FastICA(n_components=2)
    S_ = ica.fit_transform(data)  # Reconstruct signals shape (time, n_signals)
    A_ = ica.mixing_  # Get estimated mixing matrix

    return S_, A_, ica


# %% 

def plot_signal(data: np.ndarray, begin_offset_sec: float, duration: float, filtered:bool) -> None:

    # subset the data to sec seconds
    begin_idx = int(begin_offset_sec*SAMPLE_RATE)
    end_idx = int((begin_offset_sec+duration)*SAMPLE_RATE)
    data = data[begin_idx:end_idx]

    # Plot the raw and filtered data
    time_vector = np.arange(begin_offset_sec, begin_offset_sec + duration, 1/SAMPLE_RATE)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    title = 'Filtered Data' if filtered else 'Raw Data'
    ax.plot(time_vector, data)
    ax.set_title(title)
    plt.tight_layout()
    plt.show()

def plot_separations(component_data: np.ndarray, filtered: bool, component: bool, sec_to_plot:Optional[float] = 4.0) -> None:
    # component_data is 2d array, each row is a separated signal


    # Plot the raw and filtered data
    filtered_title = 'Filtered' if filtered else 'Raw'
    component_title = 'Component' if component else 'Full Reconstructed Signal'
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    end_idx = int(sec_to_plot*SAMPLE_RATE)
    # Create a time array in milliseconds
    time = np.arange(end_idx) / SAMPLE_RATE * 1000
    ax[0].plot(time, component_data[:end_idx, 0])
    ax[0].set_title(f'{component_title} 1 {filtered_title}')
    ax[0].set_ylabel('Amplitude')
    ax[0].set_xlabel('Time (ms)')
    ax[1].plot(time, component_data[:end_idx, 1])
    ax[1].set_title(f'{component_title} 2 {filtered_title}')
    ax[1].set_ylabel('Amplitude')
    ax[1].set_xlabel('Time (ms)')
    plt.tight_layout()
    plt.show()


# %% 
# The mixed signal is channel 2 (index 2) of the raw data
# It is a mixture of EMG and heart rate signals
raw_data_chan_two = raw_30k_data[2]

btype, filter = build_filter(filt_order=4, filt_range=(20, 450), fs=30000)  # replace with your actual sampling rate
filtered_data_chan_two = sosfiltfilt(filter, raw_data_chan_two)

plot_signal(raw_data_chan_two, 0, 8, filtered=False)
plot_signal(filtered_data_chan_two, 0, 8, filtered=True)


# %% Create samples out of data
window_length_sec = 8 # seconds
overlap_sec = 0# seconds

raw_samples = make_samples(raw_data_chan_two, SAMPLE_RATE, window_length_sec, overlap_sec) # (samples, window_length_points)
filtered_samples = make_samples(filtered_data_chan_two, SAMPLE_RATE, window_length_sec, overlap_sec) # (samples, window_length_points)
# %%

# Attempt to separate the mixed raw signal
# raw_samples shape (samples, time_points) HOWEVER in this case, the samples are the time points and the features are the instances of the signal
# so we need to transpose the raw_samples to be (time_points, instances_of_signal)
reconstructed, mixing_matrix, ica_raw = separate(raw_samples.T)
# reconstructed shape (n_components, time)

reconstructed_filtered, mixing_matrix_filtered, ica_filtered = separate(filtered_samples.T)

# Plot the components
plot_separations(reconstructed,filtered=False, component=True)
plot_separations(reconstructed_filtered,filtered=True, component=True)

# %%
# apply transformation to the entire signal
full_dataset_transformed = ica_raw.transform(raw_data_chan_two.reshape(-1,1))

# %%
