# DatabaseBuilder.py
"""Extracting trials from BCI2000 DAT files and saving then in a database"""

import sqlite3
from pathlib import Path
import sys
from scipy.io import loadmat
import numpy as np
import re
import csv
import datetime
import io

class DatabaseBuilder:
    """A class to build a database of all given data"""
    
    def __init__(self, path_to_all_data):
        """
        Constructor
        
        Description of table entries:
        
        'identifier' a unique integer id for each signal.
        
        'trial_id' is the unique id of the trial. Each trial generally has
        multiple signals (e.g., one per channel). All these signals share
        the same trial ID. Thus, we can use it to get signals of any specific
        trial. 
        
        'grid_id' is the unique ID of the grid. In this experiment, each grid
        includes 42 trials. We can use grid_id to extract all trials in a
        specific grid. 
        
        'data_array' is the numpy array that includes the signal. We basically
        save the entire numpy array Byte by Byte. The two convertors
        `adapt_array` and `convert_array` handle the transformation between 
        the two automatically.
        
        'stimulus_status' when ON, the stimulus was switching between
        the red and green LEDs. When OFF, the stimulus was not switching.
        In particular and based on the current set up of our experiment,
        the stimulus is either constantly ON or OFF for the first two trials 
        without any switching happening.
        
        'subject_name' the name of the participant. In this experiment, all 
        participant names start with ERP followed by exactly three digits,
        starting from ERP001.
        
        'subject_id' is the unique integer ID of each participant. This is
        basically the last three characters of the subject name converted
        into an integer. 
        
        'red' is the luminance of the red LED measured in analog-to-digital
        units (ADU).
        
        'green' is the luminance of the green LED measured in analog-to-digital
        units (ADU).
        
        'samples_count' the number of samples in the signal.
        
        'luminance_sorted' is the matrix that shows the sorted luminances 
        in the experiments. The matrix has 44 rows and two columns. The ith 
        row is the luminance of the ith trial. Each session started with two 
        steady light trials and 42 flashing light trials. Thus, we have 44
        rows. The first column is the luminance of the red LED and the second
        column is the luminance of the green LED, measured in ADU. In this 
        experiment, the luminances were sorted based on the brightness 
        (measured here as the sum of the luminances for red and green LEDs).
        Each array is saved as Bytes. 
        
        'trial_order' is an integer that determines the order of trials within
        each grid. For example, for the sixth trial, trial_order is five.
        Trial order can be inferred from luminance_sorted but it is computed
        and repeated here for convenience. 
        
        'grid_type'  is the type of the grid. In this experiment, we have
        four grid types (a, b, c, d). The luminances used in each grid type
        are different. Breaking the space into smaller chunks made data 
        collection more manageable.
        
        'experiment_date' is the date were experiments were conducted. This
        variable is extracted automatically based on the creation date of the
        .dat file that BCI2000 generates.
        
        'dat_file_id' is the unique integer ID that links the table with the
        datfiles table. This is a many to few relation. That is many trials
        share the same DAT file. 
        
        'setup_electrodes_count' is the total number of electrodes that were
        used to during the experiment. 
        
        'electrode_index' is the index of the electrode. The source code
        includes `universal_electrodes_ids` parameter that maps the electrode 
        names to IDs. Make sure to use the same parameter in all files. 
        
        'electrode_name' is the name of the electrode, e.g., Oz.
        
        'electrode_impedance' is the impedance of the electrode in kilo Ohms 
        if the electrodes were passive. For active electrodes, the impedance
        is a color, e.g., green, amber, or red.
        
        'electrode_type'  is used to determine the technology we used for the
        electrode, e.g., ActiCap active electrode.
        
        'reference' is the reference electrode.
        
        'ground'is the ground electrode
        
        'experimenter' is the name of the experimenter
        
        'room' is the location of the experiment. 
        
        'weather' is the weather outside when we were collecting data. In 
        rooms with windows, the weather could potentially affect the metamers.
        Thus, we kept documenting it. In rooms with no windows, the weather
        is irrelevant. In this cases, we use NA.
        
        'inter_stimulus_interval' is the duration of inter stimulus interval 
        in seconds, where all LEDs were off. 
        
        'trial_length' is the trial length in seconds.
        
        'target_frequency' is the frequency of alternating between the two 
        light sources in Hz. 
        
        'comment' all other comments. 
        
        Description of datfiles table columns:
        
        'dat_file_id' is a unique integer ID for each DAT file. This entry 
        matches the `dat_file_id` entry in the signals table. Using this two, 
        we can identify the DAT file for each signal. 
        
        'file' is the BCI2000 DAT file saved as a BLOB.
        
        """
        # The name of the database.
        self.database_name = "metamers_database.db"
        self.signals_table_name = "offline_grid_sampling_abcd_signals"
        self.datfiles_table_name = "offline_grid_sampling_abcd_datfiles"
        
        self.universal_electrodes_ids = {
            "O1": 1, "OZ": 2, "O2": 3, "PO7": 4,
            "PO3": 5, "POZ": 6, "PO4": 7, "PO8": 8,
            "P9": 9, "P7": 10, "P5": 11, "P3": 12, 
            "P1": 13, "PZ": 14, "P2": 15, "P4": 16, 
            "P6": 17, "P8": 18, "P10": 19, "TP9": 20,
            "TP7": 21, "CP5": 22, "CP3": 23, "CP1": 24,
            "CPZ": 25, "CP2": 26, "CP4": 27, "CP6": 28,
            "TP8": 29, "TP10": 30, "T7": 31, "C5": 32,
            "C3": 33, "C1": 34, "CZ": 35, "C2": 36, 
            "C4": 37, "C6": 38, "T8": 39, "FT9": 40,
            "FT7": 41, "FC5": 42, "FC3": 43, "FC1": 44,
            "FCZ": 45, "FC2": 46, "FC4": 47, "FC6": 48,
            "FT8": 49, "FT10": 50, "F9": 51, "F7": 52,
            "F5": 53, "F3": 54, "F1": 55, "FZ": 56,
            "F2": 57, "F4": 58, "F6": 59, "F8": 60, 
            "F10": 61, "AF7": 62, "AF3": 63, "AFZ": 64,
            "AF4": 65, "AF8": 66, "FP1": 67, "FPZ": 68,
            "FP2": 69
            }
                
        # Converts np.array to TEXT when inserting
        sqlite3.register_adapter(np.ndarray, self.adapt_array)
        
        # Converts TEXT to np.array when selecting
        sqlite3.register_converter("NUMPY_ARRAY", self.convert_array)

        # Connect to the dataset. Create it if it does not exist
        self.connection = sqlite3.connect(
           self.database_name, detect_types=sqlite3.PARSE_DECLTYPES
           )
        
        # Get the cursor. Cursor allows us to pass SQL commands.
        self.cursor = self.connection.cursor()
        
        # Create the signals and datfiles Tables
        self.create_tables()       
                
        self.identifier = 0
        self.dat_file_identifier = 0
        self.trial_id = 0
        self.grid_id = 0
        
        try:
            # path_to_all_data should point to the directory that contains
            # all data of all participants.
            self.load_all_users_data(path_to_raw_data)
            
        except sqlite3.Error as error:
            print("An error occurred: ", error)
            
        finally:
            self.cursor.close()
            self.connection.close()
        
    
    def create_tables(self):
        """Create the two tables in the database if they do not exist"""
        # Create a data table.
        try:
            
            self.cursor.execute((
               "CREATE TABLE " + self.signals_table_name + 
               """
               (
                   identifier INTEGER, 
                   trial_id INTEGER,
                   grid_id INTEGER,
                   data_array NUMPY_ARRAY,
                   stimulus_status TEXT,
                   subject_name TEXT,
                   subject_id INTEGER,
                   red INTEGER,
                   green INTEGER,
                   amber INTEGER,
                   samples_count INTEGER,
                   luminance_sorted NUMPY_ARRAY,
                   trial_order INTEGER,
                   grid_type TEXT, 
                   experiment_date TEXT,
                   dat_file_id INTEGER,
                   setup_electrodes_count INTEGER,
                   electrode_index INTEGER,
                   electrode_name TEXT,
                   electrode_impedance TEXT,
                   electrode_type TEXT,
                   reference TEXT,
                   ground TEXT,
                   experimenter TEXT,
                   room TEXT,
                   weather TEXT,
                   inter_stimulus_interval REAL,
                   trial_length REAL,
                   target_frequency REAL,
                   comment TEXT
               )
               """
               ))
            
        # If the table already exists, delete the existing one and create
        # a new one. This should be used to update a table with new data.
        # However, accidently deleting a table can lead to the loss of that
        # database. It is important to have multiple backups. 
        except sqlite3.OperationalError:            
            user_choice = input(
                f"The table {self.signals_table_name} already exists. "
                "Do you want to delete this table and re-build it "\
                + "from scratch[Y/n]? "
                )                
            
            if user_choice != "Y":
                print("User aborted execution. ")
                sys.exit()
                
            self.cursor.execute("DROP TABLE " + self.signals_table_name)
            self.cursor.execute("DROP TABLE " + self.datfiles_table_name)   

            self.cursor.execute((
               "CREATE TABLE " + self.signals_table_name + 
               """
               (
                   identifier INTEGER, 
                   trial_id INTEGER,
                   grid_id INTEGER,
                   data_array NUMPY_ARRAY,
                   stimulus_status TEXT,
                   subject_name TEXT,
                   subject_id INTEGER,
                   red INTEGER,
                   green INTEGER,
                   amber INTEGER,
                   samples_count INTEGER,
                   luminance_sorted NUMPY_ARRAY,
                   trial_order INTEGER,
                   grid_type TEXT, 
                   experiment_date TEXT,
                   dat_file_id INTEGER,
                   setup_electrodes_count INTEGER,
                   electrode_index INTEGER,
                   electrode_name TEXT,
                   electrode_impedance TEXT,
                   electrode_type TEXT,
                   reference TEXT,
                   ground TEXT,
                   experimenter TEXT,
                   room TEXT,
                   weather TEXT,
                   inter_stimulus_interval REAL,
                   trial_length REAL,
                   target_frequency REAL,
                   comment TEXT
               )
               """
               ))                    
                
                   
        self.insert_command = (            
           "INSERT INTO " + self.signals_table_name + 
           """
           (
               identifier,
               trial_id,
               grid_id,
               data_array,
               stimulus_status,
               subject_name,
               subject_id,
               red,
               green,
               amber,
               samples_count,
               luminance_sorted,
               trial_order,
               grid_type,
               experiment_date,
               dat_file_id,
               setup_electrodes_count,
               electrode_index,
               electrode_name,
               electrode_impedance,
               electrode_type,
               reference,
               ground,
               experimenter,
               room,
               weather,
               inter_stimulus_interval,
               trial_length,
               target_frequency,            
               comment
           )
           values
           (
               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
               ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
               ?, ?, ?, ?, ?, ?
           )
           """
           )
           
        # 'file' is the BCI2000 dat files. 
        self.cursor.execute((            
            "CREATE TABLE " + self.datfiles_table_name +
            """
            (
                dat_file_id INTEGER,
                file BLOB
            )
            """
            ))
        
        self.insert_datfile_command = ((            
            "INSERT INTO " + self.datfiles_table_name +
            """
            (
                dat_file_id,
                file
            )
            values
            (
                ?, ?
            )
            """
            ))
        
    def load_all_users_data(self, path_to_all_data):
        """load all users data"""
        try:
            path = Path(path_to_raw_data)
            
        except(TypeError):
            print("Error: Provided path is not a valid directory. "
                  + "Make sure the provided path actually exists. ")
            sys.exit()
        
        # Make sure the path is a director
        if not path.is_dir():
            print("Error: Provided path to the dataset is not a directory. ")
            sys.exit()
        
        # Go through every single sub-directory.
        # Discard any directory that does not have "ERP" in its name.
        for subpath in path.iterdir():
            if subpath.is_dir() and "ERP" in subpath.name:
                print("Processing " + str(subpath.absolute()))
                (data, luminance, info, subject_runs, subject_names,
                 creation_dates, dat_files) = \
                    self.load_user_data(subpath)
                               
                self.insert_user_to_database(
                    data, luminance, info, subject_runs,
                    subject_names, creation_dates, dat_files)                
            else:
                print(
                    "Skipping " + str(subpath.absolute()) 
                    + "  --> Not an ERP folder")
        
        self.connection.commit()
        self.cursor.close()
        
        return True
                    
    def insert_user_to_database(
            self, all_runs_data, luminance, info, runs, names,
            creation_dates, dat_files):
        """Add a row to the table for all signals of the given user"""
        # Get the number of existing entires in the table
        # rows = self.cursor.execute("""SELECT * FROM signals""").fetchall()
        # rows = self.cursor.execute("""SELECT * FROM datfiles""").fetchall()
        runs_count = len(all_runs_data)
        
        for data, run, dat_file in zip(
                all_runs_data, np.arange(runs_count), dat_files):   
                        
            # Insert trials one by one
            shape = (data.shape[0], data.shape[2])
            subject_name_analyzer = re.compile(r"\d\d\d")
            
            past_trial = -1
            
            for trial, electrode in np.ndindex(shape):
                
                # Increment only when trial changes
                if trial != past_trial:
                    past_trial = trial                
                    trial_id = self.trial_id
                    self.trial_id += 1
                
                current_data = data[trial, :, electrode]
                subject_name = names[run]
                creation_date = creation_dates[run]
                
                subject_number_suffix = subject_name_analyzer.search(
                    subject_name)
                
                subject_number_suffix = int(subject_number_suffix.group())
                
                red, green = luminance[run, trial, :]            
                red = int(red)
                green = int(green)
                run_info = info[run]  
                samples_count = data.shape[1]
                channels_count = data.shape[2]
                luminance_sorted = luminance[run]
                
                order = np.where(
                    (luminance_sorted == [red, green]).all(axis=1))
                
                order = int(order[0][0])
                grid_type = self.get_grid_type(luminance_sorted)
                
                if trial == 0:
                    amber = 0
                else:
                    amber = 600 
                    
                if trial == 0 or trial == 1:
                    stimulus_status = "OFF"
                else:
                    stimulus_status = "On"
                
                package = (
                    self.identifier, trial_id, self.grid_id,
                    current_data, stimulus_status,
                    subject_name, subject_number_suffix, red, 
                    green, amber, samples_count, luminance_sorted, 
                    order, grid_type, creation_date, self.dat_file_identifier,
                    channels_count                    
                    )
                
                package_info, csv_luminances = self.analyze_run_info(
                    run_info, electrode+1, subject_name)         
                
                package += package_info
                
                csv_luminances = np.array(csv_luminances)
                
                if (csv_luminances == luminance_sorted[2:]).all() == False:
                    print(
                        "Error: Luminance levels provided in the csv info " 
                        + "do not match the luminance levels in "
                        + "luminance.mat. ")
                    sys.exit()
                    
                self.cursor.execute(self.insert_command, package)
                
                self.identifier += 1
            
            self.grid_id += 1                
            package = (self.dat_file_identifier, dat_file)
            self.cursor.execute(self.insert_datfile_command, package)
            self.dat_file_identifier += 1
    
    def get_grid_type(self, luminance):
        """Given the luminances find the grid type"""
        red = luminance[2, 0]
        green = luminance[2, 1]
        
        if red == 0 and green == 0:
            return "a"
        
        if red == 24 and green == 12:
            return "b"
        
        if red == 24 and green == 0:
            return "c"
        
        if red == 0 and green == 12:
            return "d"
        
        print("Error: Unknown grid type. ")
        sys.exit()
         
    def load_user_data(self, path):
        """Load all data in the provided directory"""
        # We must first collect all data so we can pick a common trial
        # length for all of them. 
        signals = []
        states = []
        luminance = []
        subject_names = []
        info = []       
        creation_dates = []
        dat_files = []
        
        runs_count = self.get_number_of_runs(path)        
        subject_name_analyzer = re.compile(r"ERP(\d\d\d)_(\d\d)")        
                    
        # iterdir does not guarantee any sepcific order.
        # We must ensure we read the files in order.
        # This is not a very efficient way to guarantee this but it works.
        for current_run in np.arange(1, runs_count+1):
            
            for subpath in path.iterdir():    
                subject_info = subject_name_analyzer.search(subpath.name)
                
                if subject_info is None:
                    continue                
                
                # Skip hidden files
                if subpath.name[0] == '.':
                    continue
                
                subject_name = "ERP" + subject_info.group(1)
                subject_run = int(subject_info.group(2))
                  
                # The following IF statement and the outter FOR loop ensure
                # we only read one trial at a file lest we mix files from
                # different trials.
                if subject_run != current_run:
                    continue
                                    
                if "signal" in subpath.name and subpath.suffix == ".mat":                
                    signals.append(self.load_signal(subpath))
                if "states" in subpath.name and subpath.suffix == ".mat":         
                    states.append(self.load_states(subpath))
                if "luminance" in subpath.name and subpath.suffix == ".mat":
                    luminance.append(self.load_luminance(subpath))  
                if "info" in subpath.name and subpath.suffix == ".csv":
                    info.append(self.load_info(subpath))
                if subpath.suffix == ".dat":
                    date = subpath.stat().st_mtime
                    date = datetime.datetime.fromtimestamp(date)
                    date = str(date)
                    creation_dates.append(date)
                    subject_names.append(subject_name)
                    dat_file = self.load_dat_file(subpath)
                    dat_files.append(dat_file)
                
        data = self.extract_all_trials(signals, states)
        luminance = np.array(luminance)
        subject_runs = np.arange(runs_count)
        
        return (
            data, luminance, info,
            subject_runs, subject_names, 
            creation_dates, dat_files)
    
    def get_number_of_runs(self, path):
        """Get how many runs there are in each folder"""
        subject_runs = []
        subject_name_analyzer = re.compile(r"ERP(\d\d\d)_(\d\d)")
        
        for subpath in path.iterdir():
            subject_info = subject_name_analyzer.search(subpath.name)
            
            if subject_info is not None and subpath.suffix == ".dat":               
                subject_runs.append(int(subject_info.group(2)))
                
        return len(subject_runs)
        
       
    def load_signal(self, path):
        """
        Load signal from the path.
        'signal' contains the actual EEG data.
        """
        try:
            signal = loadmat(str(path.absolute()))
            
        except(FileNotFoundError):
            print(
                "Could not laod the signal file.  No file at " 
                + str(path.absolute()) + ". "
                )
            sys.exit()
            
        return signal['signal']
    
    def load_states(self, path):
        """
        Load states from the path.
        'states' includes when stimulator was on or off.
        We need it to extract trials from the signal.
        """
        try:
            states = loadmat(str(path.absolute()))
        except(FileNotFoundError):
            print(
                "Could not laod the states file.  No file at " 
                + str(path.absolute()) + ". "
                )
            sys.exit()
        
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
            print("Luminance levels file has not attribute stimValsSorted.  ")
            sys.exit()
            
        # The first two trials are always fixed.
        # The first trial is all LEDs off
        # The second trial is all LEDs on
        # We mannually insert them here. 
        luminance_level = np.zeros((luminance.shape[0]+2, luminance.shape[1]))
        luminance_level[1] = [600, 600]
        luminance_level[2:] = luminance
        
        return np.int32(luminance_level)
    
    def load_dat_file(self, path):
        """Load and return the data file"""
        try:
            dat_file = open(path, mode='r+b')
        except(FileNotFoundError):
            self.quit(
                "Could not find the BCI2000 dat file at "
                + str(path.absolute()) + ". "
                )
            
        dat_file_blob = dat_file.read()
  
        return dat_file_blob
    
    def load_info(self, path):
        """Read the CSV file that contains experiment infos"""
        info_lines = []
        try:            
            reader = csv.reader(open(path, "rt"), delimiter=",")
            mycsv = open(path, "rt")
            reader = csv.reader(x.replace('\0', '') for x in mycsv)
            
        except(FileNotFoundError):
            self.quit(
                "Could not load the info file.  No file at "
                + str(path.absolute()) + ". "
                )        

    
        for row in reader:
            info_lines.append(row)
            
        return info_lines
        
      
    def extract_all_trials(self, signals, states):
        """Extract equal-size trials from all runs"""        
        trials = []
        lengths = []
        
        for signal, state in zip(signals, states):
            tokenized_signal = self.extract_trials(signal, state)
            trials.append(tokenized_signal)
            lengths.append(tokenized_signal.shape[1])
            
        doable_length = min(lengths)
        
        truncated_trials = []
        
        for trial in trials:
            truncated_trials.append(trial[:, :doable_length, :])
            
        # signal_final = np.array(truncated_trials)
        # return signal_final  
        return truncated_trials
        
    def extract_trials(self, signal, state):
        """Extract trials from a single run"""
        # Find the locations where the stimulus transitions from on to off
        # or off to on
        edges = state[1:] - state[:-1]
        edges = np.insert(edges, 0, 0)
        
        # Positive edges represent a transition from off to on
        # Negative edges represent a transition from on to off
        positive_edges = np.where(edges == 1)[0]
        negative_edges = np.where(edges == -1)[0]
        
        # For each trial, find for how many samples the stimulus was off
        # and for how many samples the stimulus was on.
        # These durations vary from trials to trials by a few samples.
        off_durations = positive_edges[1:] - negative_edges[:-1]
        on_durations = negative_edges - positive_edges
        
        # Pick the shortest durations. This allows us to have equal size
        # trials. We are throwing away only a few samples. 
        doable_off_duration = np.min(off_durations)
        doable_on_duration = np.min(on_durations)
        
        trials_count = positive_edges.size
        electrodes_count = signal.shape[1]
        
        tokenized_signal = np.zeros(
            (trials_count, doable_off_duration+doable_on_duration,
             electrodes_count))
        
        # Extract trials. 
        for i in np.arange(positive_edges.size):
            index = positive_edges[i]
            tokenized_signal[i] = signal[
                index-doable_off_duration:index+doable_on_duration]
            
        return tokenized_signal
    
    def analyze_run_info(self, info, electrode_index, subject_name):
        """Extract trial info"""
        comment = None
        weather = None
        room = None
        experimenter = None
        gnd = None
        ref = None
        isi = None
        trial_length = None
        electrode_name = None
        electrode_impedance = None
        electrode_type = None
        csv_luminances = []
        selected_index = None
        
        electrode_name_processor = re.compile(r"electrode-(\d+)")
        
        luminance_processor = re.compile(
            r"round (\d+)/(\d+) - (\d+)red, (\d+)green")
        
        for line in info:
            
            # Skip blank lines
            if len(line) == 0:
                continue
            
            attribute = line[0].lower().strip()
            
            if "bci2000 parameter file:" in attribute:
                break
                        
            content_1 = line[1].strip()
            
            if len(line) == 3:
                content_2 = line[2].strip()
            
            if content_1 is None:
                content_1 = "None"
            
            if content_2 is None:
                content_2 = "None"
                
            content_1 = content_1.lower()
            content_2 = content_2.lower()
                
            if "comment" in attribute:
                comment = content_1
            elif "weather" in attribute:
                weather = content_1
            elif "room" in attribute:
                room = content_1
            elif "experimenter" in attribute:
                experimenter = content_1
            elif "gnd location" in attribute:
                gnd = content_1
            elif "ref location" in attribute:
                ref = content_1
            elif "inter stimulus interval" in attribute:
                isi = float(content_1)
            elif "trial length" in attribute:
                trial_length = float(content_1)
            elif "target frequency" in attribute:
                target_frequency = float(content_1)
            elif "date" in attribute:
                continue
            elif "participant" in attribute:
                participant_name = content_1
                
                if participant_name != subject_name.lower():
                    print(
                        "Error: Participant name in the info.csv does not "
                        + "match the file and folder name. ")
                    sys.exit()
                    
            elif "round" in attribute:
                combined_line = attribute + ", " + content_1
                luminance = luminance_processor.search(combined_line)
                red = int(luminance.group(3))
                green = int(luminance.group(4))
                csv_luminances.append([red, green])

            elif "electrode-" in attribute:               
                electrode_id = electrode_name_processor.search(attribute)                
                electrode_id = electrode_id.group(1)
                electrode_id = int(electrode_id)
                
                if electrode_id != electrode_index:
                    continue
                
                electrode_name = content_1.upper()
                electrode_impedance = content_2
                
                # The impedance for the active electrodes is not 
                # numeric. Instead it is green, amber, red, etc. 
                try:
                    float(content_2)
                    electrode_type = "EasyCAP CAP and Electrode"
                except(ValueError):
                    electrode_type = "ActiCap Active Electrode"
                    
                electrode_name = electrode_name.upper()
            
                try:
                    selected_index = \
                        self.universal_electrodes_ids[electrode_name]                
                except(KeyError):
                    print(
                        f"Error: Undefined electrode. {electrode_name} in the "
                        + "info csv file is not in 10-10 international "
                        + "system. ")
                    sys.exit()          
                     
            elif len(attribute) == 0 and len(content_1) == 0:
                continue
            
            else:
                print(
                    "Error: The info.csv file contains undefined entries. "
                    + "Ensure the file is formatted correctly. ")
                sys.exit()
            
        if gnd in self.universal_electrodes_ids.keys():
            gnd = gnd.upper()

        package = (
            selected_index, electrode_name, electrode_impedance,
            electrode_type, ref.upper(), gnd, experimenter, room, weather,
            isi, trial_length, target_frequency,
            comment
            )
        
        for p in package:
            if p is None:
                print(
                    "Error: The info csv file has some missing "
                    + "information. Make sure it is complete. ")
                sys.exit()
                
        return package, csv_luminances
    
    def adapt_array(self, numpy_array):
        """Convert an array into bytes so we can save in the database"""  
        out = io.BytesIO()
        np.save(out, numpy_array)
        out.seek(0)
        return sqlite3.Binary(out.read())

    def convert_array(self, text):
        """Convert blob to numpy array"""
        out = io.BytesIO(text)
        out.seek(0)
        return np.load(out)

if __name__ == '__main__':    
    path_to_raw_data = r"D:\Research\Datasets\MetamersDataset\AbcdGridsDataset"
    builder = DatabaseBuilder(path_to_raw_data)
    