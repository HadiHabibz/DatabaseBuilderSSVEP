# SSVEP Database Builder
This implementation includes the python code to automatically read BCI2000 DAT files and save them in a database.

Saving all signals in a database has a number of advantages:
1. No information about the experiments will be lost.
2. The process is automatic. The streamer file used for data collection generates all files automatically with a format that is compatible with this database builder. If the user changes settings or order of parameters, these two pieces of code (streamer and database builder) automatically handle the differences. 
3. We can easily run queries to get the data we want quickly.
4. We do not need to worry about aligning trials with triggers and timing issues. 

# Database Structure
The database has a very simple (and most likely not so optimized) structure. Each experiment has two tables. 

- experiment_name_signals: This table includes the actual signals and corresponding parameters. The experiment_name is the name of the experiment. The signals are stored as MAT files. 
- experiment_name_datfiles: This table holds the BCI2000 dat files. 

The structure of each table varies from experiment to experiment. Typically, you will only need to deal with the signals table. The datfiles table is included for sake of completeness. It is recommended to keep the structure of the tables intact as much as possible to ensure compatibility among different scripts. 

# Experiment 1: offline_grid_sampling_abcd

In this experiment, we wanted to obtain high-resolution grids for our participants. We sampled the grid at 168 locations. At each location, we collected seven seconds of data. Six seconds of stimulation followed by one second of inter-stimulation interval. We collected data 42 locations at time, with break sessions in between for the participant to rest. 

## The Structure of the `signals` Table:
The signals table has the following columns:
- `identifier` is a unique integer id for each signal.
- `trial_id` is the unique integer id of the trial. Each trial generally has multiple signals (e.g., one per channel). All these signals share the same trial ID. Thus, we can use it to get signals of any specific trial. 
- `grid_id`  is the unique integer ID of the grid. In this experiment, each grid includes 42 trials. We can use grid_id to extract all trials in a specific grid. 
- `data_array` is the numpy array that includes the signal. We basically save the entire numpy array Byte by Byte. The two convertors `adapt_array` and `convert_array` handle the transformation between the two automatically.
- `stimulus_status`  when ON, the stimulus was switching between the red and green LEDs. When OFF, the stimulus was not switching.  In particular and based on the current set up of our experiment, the stimulus is either constantly ON or OFF for the first two trials without any switching happening.
- `subject_name`the name of the participant. In this experiment, all subject names start with ERP followed by exactly three digits, starting from ERP001.
- `subject_id` is the unique integer ID of each participant. This is basically the last three characters of the subject name converted into an integer. 
- `red` is the luminance of the red LED measured in analog-to-digital units (ADU).
- `green` is the luminance of the green LED measured in analog-to-digital units (ADU).
- `samples_count` is the number of samples in the signal.
- `luminance_sorted` is the matrix that shows the sorted luminances in the experiments. The matrix has 44 rows and two columns. The ith row is the luminance of the ith trial. Each session started with two steady light trials and 42 flashing light trials. Thus, we have 44 rows. The first column is the luminance of the red LED and the second column is the luminance of the green LED, measured in ADU. In this experiment, the luminances were sorted based on the brightness (measured here as the sum of the luminances for red and green LEDs). Each array is saved as Bytes. 
- `trial_order` is an integer that determines the order of trials within each grid. For example, for the sixth trial, trial_order is five. Trial order can be inferred from luminance_sorted but it is computed and repeated here for convenience. 
- `grid_type` is the type of the grid. In this experiment, we have four grid types (a, b, c, d). The luminances used in each grid type are different. Breaking the space into smaller chunks made data collection more manageable.
- `experiment_date` is the date were experiments were conducted. This variable is extracted automatically based on the creation date of the DAT file that BCI2000 generates.
- `dat_file_id` is the unique integer ID that links the table with the datfiles table. This is a many to few relation. That is many trials share the same DAT file. 
- `setup_electrodes_count` is the total number of electrodes that were used to during the experiment. 
- `electrode_index` is the index of the electrode. The source code includes `universal_electrodes_ids` parameter that maps the electrode names to IDs. Make sure to use the same parameter in all files. 
- `electrode_name`is the name of the electrode, e.g., Oz.
- `electrode_impedance` is the impedance of the electrode in kilo Ohms if the electrodes were passive. For active electrodes, the impedance  is a color, e.g., green, amber, or red.
- `electrode_type` is used to determine the technology we used for the electrode, e.g., ActiCap active electrode.
- `reference` is the the reference electrode.
- `ground`  is the ground used for measuring EEG.
- `experimenter` is  the name of the experimenter.
- `room` is the location of the experiment. 
- `weather` is the weather outside when we were collecting data. In rooms with windows, the weather could potentially affect the metamers. Thus, we kept documenting it. In rooms with no windows, the weather is irrelevant. In this cases, we use none.
- `inter_stimulus_interval`is the duration of inter stimulus interval in seconds, where all LEDs were off. 
- `trial_length` is the length of the trial in seconds. 
- `target_frequency` is the frequency of alternating between the two light sources in Hz. 
- `comment` is the experimenter's comments. 

## The Structure of the `datfiles` Table:
This table holds the BCI2000 dat files. There are only two columns:

- `dat_file_id` is a unique integer ID for each DAT file. This entry matches the `dat_file_id` entry in the signals table. Using this two, we can identify the DAT file for each signal. 
- `file` is the BCI2000 DAT file saved as a BLOB.

## Experiment Comments and Notes

We moved the rooms a few times. Initially, we were located in A625. We, then moved to A629. We finally moved one more time to A635. Each of these rooms had it own lighting conditions that could affect the data. 
We also changed the electrode technology. About one quarter of our data is collected with the passive EasyCap electrodes. We, then, switched to the ActiCap active electrodes. Also, at some point we switched from 16 channels to 32 and change the order and location of the electrodes on the cap. All these changes are documented in the base. Also, the database is immune to changes in the cap or order of electrodes.  
