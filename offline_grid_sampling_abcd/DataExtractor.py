# DataExtractor.py
"""A class to extract trials and relevant info from metamers data files"""

import FeatureExtractorSSVEP
import numpy as np
from scipy.io import loadmat
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import numpy.ma as ma

class DataExtractor:
    """
    Read data and other important signals
    """
    def __init__(self, path_to_data):
        """
        Create and extractor object        
        """
        self.figsize = (8, 8)
        self.data, self.luminance = self.load_data(path_to_data)   
        self.data = np.transpose(self.data, axes=(0, 1, 3, 2))
        self.get_reds_and_greens()  
        
        # Drop the first two trials.
        # These are always on and always off trials.        
        self.data = self.data[:, 2:, :, :]
        self.luminance = self.luminance[:, 2:, :]
        
    def load_data(self, path_to_data):
        """
        Load .m data and flags
        """
        # Convert the string path to Path
        try:
            path = Path(path_to_data)
        except(TypeError):
            self.quit("The provided path is not a valid directory name. ")
            
        if path.is_dir() == False:
            self.quit("The provided path is not a direcotory. ")
            
        signals = []
        states = []
        luminance = []
            
        for subpath in path.iterdir():            
            if "signal" in subpath.name and subpath.suffix == ".mat":                
                signals.append(self.load_signal(subpath))
            if "states" in subpath.name and subpath.suffix == ".mat":         
                states.append(self.load_states(subpath))
            if "luminance" in subpath.name and subpath.suffix == ".mat":
                luminance.append(self.load_luminance(subpath))                
                
        data = self.extractAllTrials(signals, states)
        luminance = np.array(luminance)

        return (data, luminance)
       
    def load_signal(self, path):
        """
        Load signal from the path.
        'signal' contains the actual EEG data.
        """
        try:
            signal = loadmat(str(path.absolute()))
        except(FileNotFoundError):
            self.quit(
                "Could not laod the signal file.  No file at " 
                + str(path.absolute()) + ". "
                )
            
        return signal['signal'][:, :16]
    
    def load_states(self, path):
        """
        Load states from the path.
        'states' includes when stimulator was on or off.
        We need it to extract trials from the signal.
        """
        try:
            states = loadmat(str(path.absolute()))
        except(FileNotFoundError):
            self.quit(
                "Could not laod the states file.  No file at " 
                + str(path.absolute()) + ". "
                )
        
        # Ensure states are integers
        return np.int32(states['states'])
    
    def load_luminance(self, path):
        """
        load luminance levels.
        'luminance' includes the red and green values associated with 
        each signal. 
        """
        try:
            luminance = loadmat(str(path.absolute()))
        except(FileNotFoundError):
            self.quit(
                "Could not load the luminance levels file.  No file at "
                + str(path.absolute()) + ". "
                )
            
        try:
            luminance = luminance['stimValsSorted']
        except(KeyError):
            self.quit("Luminance levels file has not attribute stimValsSorted")
            
        # The first two trials are always fixed.
        # The first trial is all LEDs off
        # The second trial is all LEDs on
        # We mannually insert them here. 
        luminance_level = np.zeros((luminance.shape[0]+2, luminance.shape[1]))
        luminance_level[1] = [600, 600]
        luminance_level[2:] = luminance
        
        return np.int32(luminance_level)
        
    def extractAllTrials(self, signals, states):
        """Extract equal-size trials from all runs"""        
        trials = []
        lengths = []
        
        for signal, state in zip(signals, states):
            tokenizedSignal = self.extractTrials(signal, state)
            trials.append(tokenizedSignal)
            lengths.append(tokenizedSignal.shape[1])
            
        doableLength = min(lengths)
        
        truncatedTrials = []
        
        for trial in trials:
            truncatedTrials.append(trial[:, :doableLength, :])
            
        signalFinal = np.array(truncatedTrials)
        return signalFinal            
        
    def extractTrials(self, signal, state):
        """Extract trials from a single run"""
        # Find the locations where the stimulus transitions from on to off
        # or off to on
        edges = state[1:] - state[:-1]
        edges = np.insert(edges, 0, 0)
        
        # Positive edges represent a transition from off to on
        # Negative edges represent a transition from on to off
        positiveEdges = np.where(edges == 1)[0]
        negativeEdges = np.where(edges == -1)[0]
        
        # For each trial, find for how many samples the stimulus was off
        # and for how many samples the stimulus was on.
        # These durations vary from trials to trials by a few samples.
        offDurations = positiveEdges[1:] - negativeEdges[:-1]
        onDurations = negativeEdges - positiveEdges
        
        # Pick the shortest durations. This allows us to have equal size
        # trials. We are throwing away only a few samples. 
        doableOffDuration = np.min(offDurations)
        doableOnDuration = np.min(onDurations)
        
        trialsCount = positiveEdges.size
        electrodesCount = signal.shape[1]
        
        tokenizedSignal = np.zeros(
            (trialsCount, doableOffDuration+doableOnDuration, electrodesCount))
        
        # Extract trials. 
        for i in np.arange(positiveEdges.size):
            index = positiveEdges[i]
            tokenizedSignal[i] = signal[
                index-doableOffDuration:index+doableOnDuration]
            
        return tokenizedSignal
    
    def get_reds_and_greens(self):
        """Get the unique luminance levels of red and green leds"""
        luminance = self.luminance
        luminance = luminance[:, 2:]
        luminance = np.reshape(luminance, (-1, 2))
        self.reds = np.unique(luminance[:, 0])
        self.greens = np.unique(luminance[:, 1])
                                      
    def extract_features(self, targets_frequencies, subbands):
        """Extract features using common feature extraction methods"""        
        harmonics_count = 4
        sampling_rate = 256
        filter_order = 12
        
        extractor = [
            FeatureExtractorSSVEP.FeatureExtractorMEC(),
            FeatureExtractorSSVEP.FeatureExtractorMSI(),
            FeatureExtractorSSVEP.FeatureExtractorCCA(),
            ]
        
        # extractor = [
        #     None,
        #     None,
        #     FeatureExtractorSSVEP.FeatureExtractorCCA(),
        #     ]
            
        # CCA does not scale with the number of cores.
        # Thus, the last one is set to zero.
        core_counts = [8, 8, 8]
        
        # Make data 3D. This is a requirement for feature extractors. 
        data = np.reshape(self.data, (
            -1, self.data.shape[-2], self.data.shape[-1]))
        
        # Discard the first second of data.
        # This is ISI and shount not be included in the data analysis
        data = data[:, :, 256:]
            
        # Create a list for the results
        self.features = [0] * 6
        index = 0
        
        for phi, cores in zip(extractor, core_counts):
            
            if phi is None:
                self.features[index]  = None
                self.features[index+3] = None
                index += 1
                continue
            
            phi.setup_feature_extractor(
            harmonics_count=harmonics_count,
            targets_frequencies=targets_frequencies,
            sampling_frequency=sampling_rate,          
            use_gpu=False,
            explicit_multithreading=cores,
            max_batch_size=16,
            filter_order=filter_order,
            subbands=subbands,
            samples_count=data.shape[-1])
            
            features = phi.extract_features(data)[:, :, 0, :, 0]
            
            features = np.reshape(
                features, self.data.shape[:2] + features.shape[1:])
            
            # The first subband is used as the non-filterbank variant.
            self.features[index] = features[:, :, 0, :]
            
            # All subbands are used for filter bank variants.
            self.features[index+3] = features[:, :, :, 0]
            
            index += 1
        
    def extract_sub_grids(self, subgrid_size=42):
        """
        Extract subgrids.
        This function can be used to break up super grids to their 
        constituting subgrids. 
        """       
        samples_count = self.luminance.shape[1]
        subgrid_count = samples_count // subgrid_size
        sessions_count = self.luminance.shape[0]            
        index_tracker = 0
        
        for grid in self.features:
            self.features[index_tracker] = np.reshape(
                grid, (sessions_count, subgrid_count, subgrid_size, -1))
            index_tracker += 1
            
        self.luminance = np.reshape(
            self.luminance, (sessions_count, subgrid_count, subgrid_size, -1))
        
    def merge_sub_grids(self):
        """
        Merge subgrids.
        This function can be used to undo extract_sub_grids.
        """
        sessions_count = self.luminance.shape[0]               
        index_tracker = 0
        
        for grid in self.features:
            features_count = grid.shape[-1]
            self.features[index_tracker] = np.reshape(
                grid, (sessions_count, -1, features_count))
            index_tracker += 1
            
        self.luminance = np.reshape(
            self.luminance, (sessions_count, -1, 2))
        
        
    def merge_grids(self, to_merge_tuple):
        """
        Merge the current data with the given data.
        Use masked values to incidate missing content. 
        """   
        
        # Createa a tuple of all grids to be merged. 
        grids_tuples = to_merge_tuple + (self,)
        
        runs_count = self.data.shape[0]
        total_trials_count = 0
        electrodes_count = self.data.shape[2]
        all_samples_count = []        
        
        # Make sure the sizes are consistent. All grids must have the 
        # same number of electrodes and the same number of samples. 
        # For example, we can merge two grids, where the first one has
        # six runs while the second one has four runs. 
        for grid in grids_tuples:
            
            total_trials_count += grid.data.shape[1]
            
            if electrodes_count != grid.data.shape[2]:
                self.quit("Could not merge data with different number " 
                         + "of electrodes. ")
                
            if runs_count != grid.data.shape[0]:
                self.quit("Could not merge data with different number "
                          + "of runs. ")
                
            all_samples_count.append(grid.data.shape[-1])
            
        samples_count = np.min(all_samples_count)
            
        data = np.zeros((
            runs_count, 
            total_trials_count,            
            electrodes_count,
            samples_count
            ), dtype=np.float32)
         
        luminance = np.zeros((
            runs_count,
            total_trials_count,
            2))
        
        trial_index = 0
        
        for grid in grids_tuples:
            data[:, trial_index:trial_index+grid.data.shape[1]] =\
                grid.data[:, :, :, :samples_count]
            
            luminance[:, trial_index:trial_index+grid.data.shape[1]] =\
                grid.luminance
        
            trial_index += grid.data.shape[1]
        
        reds = np.unique(luminance[0, :, 0])
        greens = np.unique(luminance[0, :, 1])
        
        self.luminance = luminance
        self.data = data
        self.reds = reds
        self.greens = greens
        
          
    def plot_average_grids(self, features=None):
        """Plot the grid data"""
        if features is None:
            features = self.features
        
        # We only need this for aggregating filter bank results 
        featurizer = FeatureExtractorSSVEP.FeatureExtractorCCA()
        
        for f in features:
            
            if f is None:
                continue
            
            f = featurizer.filterbank_standard_aggregator(f, axis=2)
            f = np.mean(f, axis=0)           
            self.plot_meshgrid(f)
                                 
    def plot_meshgrid(self, data):
        """Plot the given 2D matrix as a mesh"""
        fig, axis = plt.subplots(figsize=self.figsize)
        # self.center_grids()
        data_grid = self.reshape_features_to_grid(data, self.luminance[0])
        red_ticks = np.arange(0, len(self.reds)) + 0.5
        green_ticks = np.arange(0, len(self.greens)) + 0.5
        size = np.asarray(data_grid.shape) + 1
        x = np.zeros(size)
        y = np.zeros(size)
        
        for i, j in np.ndindex(tuple(size)):
            x[i, j] = i
            y[i, j] = j
        
        axis.pcolormesh(y, x, data_grid, shading="flat")
        
        axis.set_yticks(red_ticks)
        axis.set_yticklabels(self.reds)
        axis.set_xticks(green_ticks)
        axis.set_xticklabels(self.greens)
        axis.set_ylabel("Red")
        axis.set_xlabel("Green")
        
        plt.show()
        # import datetime
        # fig.savefig("grid_dense" + str(datetime.datetime.now()) + ".pdf")
        return (fig, axis)
    
    def get_grid_view(self):
        """Convert all features to the grid view"""
        features = []
        luminance = []
        
        for i in np.arange(len(self.features)):
            f, l = self.reshape_features_to_grid(
                self.features[i], self.luminance)
            features.append(f)
            luminance.append(l)
            
        return (features, luminance)
    
    def reshape_features_to_grid(self, data=None, luminance=None):
        """Reshape the 1D data into a grid structure"""
        grid = np.zeros((self.reds.size, self.greens.size))
        masks = np.zeros((self.reds.size, self.greens.size))
               
        for r_index, g_index in np.ndindex((self.reds.size, self.greens.size)):
            r = self.reds[r_index]
            g = self.greens[g_index]
            
            matched = np.where((luminance == [r, g]).all(axis=1))[0]
            
            if matched.size == 0:
                masks[r_index, g_index] = 1
            else:            
                grid[r_index, g_index] = data[matched]
            
        grid_masked = ma.array(grid, mask=masks) 
        return grid_masked    
       
    def center_grids(self):
        """De-mean all grids"""
        self.extract_sub_grids()       
        index = 0
        
        for grid in self.features:
            self.features[index] = (
                self.features[index] 
                - np.mean(self.features[index], axis=2)[:, :, None, :]
                )
            index += 1
            
        self.merge_sub_grids()
        
    def quit(self, message="unknown error. "):
        """Quit the execution"""
        print("Error: " + message)
        sys.exit()
        
               

subbands = np.array([[i*5, 55] for i in range(1, 8)])
targets_frequencies = 10

extractor = DataExtractor(
    r"C:\Users\Hadi\Downloads\Metamers_Dataset_Round_2\ERP007a")

extractor1 = DataExtractor(
    r"C:\Users\Hadi\Downloads\Metamers_Dataset_Round_2\ERP007b")

extractor2 = DataExtractor(
    r"C:\Users\Hadi\Downloads\Metamers_Dataset_Round_2\ERP007c")

extractor3 = DataExtractor(
    r"C:\Users\Hadi\Downloads\Metamers_Dataset_Round_2\ERP007d")

extractor.merge_grids((extractor1, extractor2, extractor3))
extractor.extract_features(targets_frequencies, subbands)
extractor.plot_average_grids()
