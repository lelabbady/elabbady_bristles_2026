import pandas as pd
import json
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import cdist


leg_tips = ['L1D_pt_position','L1E_pt_position',
            'L2D_pt_position','L2E_pt_position',
            'R1D_pt_position','R1E_pt_position',
            'R2D_pt_position','R2E_pt_position',]

def velo_swing_stance(df, joint, parse_by='fullfile', 
                      legs=['L1','R1','L2','R2','L3','R3'], 
                      velo_thresh=0, lower_velo_thresh=-35, 
                      dt=1/300):
    
    print('classifying swing and stance...')
    for leg in legs:
        total_step_num = 0
        for bout in tqdm(df[parse_by].unique()):
            check = df[parse_by] == bout
            
            # calculate leg tip velocity relative to body-coxa joint
            leg_velo = (np.sqrt(np.diff(df.loc[check, leg+f'{joint}_x'])**2 +\
                                np.diff(df.loc[check, leg+f'{joint}_y'])**2 +\
                                np.diff(df.loc[check, leg+f'{joint}_z'])**2) \
                       )/dt
            leg_norm_y = df.loc[check,leg+f'{joint}_y'].values - df.loc[check,leg+'A_y'].values
            leg_velo[np.diff(leg_norm_y)<0] = -leg_velo[np.diff(leg_norm_y)<0]
            
            df.loc[check, leg+'_velo'] = np.insert(leg_velo, 0, leg_velo[0])
        
            s=0.5 # smoothing parameter
            leg_smoothed_velo = gaussian_filter1d(leg_velo, s)
            df.loc[check, leg+f'_{joint}_smoothed_velo'] = np.insert(leg_smoothed_velo, 0, leg_smoothed_velo[0])
            
            # determine swing and stance based on velocity threshold
            swing = (df[leg+f'_{joint}_smoothed_velo'] > velo_thresh) | (df[leg+f'_{joint}_smoothed_velo'] <= lower_velo_thresh)
            stance = df[leg+f'_{joint}_smoothed_velo'] <= velo_thresh
            df.loc[check & swing, leg+f'_{joint}_swing_stance'] = 0
            df.loc[check & stance, leg+f'_{joint}_swing_stance'] = 1
            
            swing_stance = df.loc[check, leg+f'_{joint}_swing_stance'].to_numpy()
            for i in [3,4,5,3]:
                for j in range(len(swing_stance)-(i-1)):
                    if swing_stance[j] == swing_stance[j+(i-1)] and swing_stance[j+1] != swing_stance[j] and np.all(swing_stance[j+1:j+(i-1)]== swing_stance[j+1]):
                        swing_stance[j+1:j+(i-1)] = swing_stance[j]
            df.loc[check, leg+f'_{joint}_swing_stance'] = swing_stance
            
            # Initialize step number and previous value variables
            prev = df.loc[check, leg+f'_{joint}_swing_stance'].iloc[0]
            step_num = 0 if prev == 0 else 1

            # Create a list to store step numbers
            step_num_arr = [step_num]

            # Iterate through the DataFrame and assign step numbers for the bout
            for cur in df.loc[check, leg+f'_{joint}_swing_stance'].iloc[1:]:
                #print(prev)
                if prev == 0 and cur == 1:
                    step_num += 1

                step_num_arr.append(step_num)
                prev = cur
            step_num_arr = np.array(step_num_arr)
        
            # Assign the step numbers as a new column
            df.loc[check, leg+f'_{joint}_bout_step_num'] = step_num_arr
            
            # Keep track of step start and end indices
            min_max = df.loc[check].groupby(f'{leg}_{joint}_bout_step_num').index.agg(['min', 'max'])
            for step in min_max.index:
                is_step = df[leg+'_D_bout_step_num'] == step #steps are 1 indexed, iterator is 0 indexed
                df.loc[check & is_step, leg+f'_{joint}_bout_step_start'] = min_max['min'][step]
                df.loc[check & is_step, leg+f'_{joint}_bout_step_end'] = min_max['max'][step]
                                  
    return df

def at_least_two_meet_threshold(value1, value2, value3, threshold):
    """
    Checks if at least two of the three input values meet or exceed the given threshold.

    Args:
        value1 (float): The first value.
        value2 (float): The second value.
        value3 (float): The third value.
        threshold (float): The threshold to check against.

    Returns:
        bool: True if at least two values meet or exceed the threshold, False otherwise.
    """
    count = sum(val >= threshold for val in [value1, value2, value3])
    return count >= 2

def preprocess_data(df):
    '''Preprocesses the input DataFrame by adding additional columns for analysis.
    df (pandas.DataFrame): 
    Outputs:
    The processed DataFrame with additional columns:
        - 'repnum': The repetition number extracted from the 'fullfile' column.
        - 'fly': The fly identifier extracted from the 'flyid' column.
        - 'laser': A binary value indicating laser status based on the 'fnum' column.
        - 'genotype': The genotype of the fly mapped from a JSON file.'''
    
    df['repnum'] = [i.split('R')[-1] for i in df.fullfile]
    df['fly'] = df['flyid'].str.split('_').str[-2]
    df['fly'] = df['fly'].str.split(' ').str[-1]
    #df['laser'] = (df.fnum // 1500) % 2

    # import fly genotypes 
    with open('../data/fly_genotypes.json', 'r') as f:
        fly_dict = json.load(f)
    df['genotype'] = df.flyid.map(fly_dict)
    print('added genotypes')

    # import distance thresholds tuned per fly
    with open('../data/fly_thresholds.json', 'r') as f:
        tresh_dict = json.load(f)
    df['threshold'] = df.flyid.map(tresh_dict)
    print('added thresholds')

    #calculate velocities of FeTi 'C' and TiTa 'D' joints, Tarsi tip automatically calculated
    df = velo_swing_stance(df, joint = 'D')
    df = velo_swing_stance(df, joint = 'C')

    # Calculate the derivative for every column in df that ends with '_smoothed_velo'
    velo_columns = [col for col in df.columns if col.endswith('_smoothed_velo')]
    temp = pd.DataFrame()
    for col in velo_columns:
        derivative_col = f"{col}_d1"
        temp[derivative_col] = df[col].diff() / df['fnum'].diff()
        temp[derivative_col] = temp[derivative_col].fillna(0).abs()
    
    df = pd.concat([df, temp], axis=1)

    return df

def get_pt_positions(df):
    ''''Extracts and calculates 3D positions for specific body parts (joints) of the fly.
    Inputs:
        - df (pandas.DataFrame): The input DataFrame containing joint coordinate columns.
    Outputs:
        - pandas.DataFrame: The DataFrame with additional columns for each joint's 3D position.
                     Each new column contains a numpy array representing the 3D coordinates.'''
    
    xs = ['A_x', 'B_x', 'C_x', 'D_x', 'E_x']
    ys = ['A_y', 'B_y', 'C_y', 'D_y', 'E_y']
    zs = ['A_z', 'B_z', 'C_z', 'D_z', 'E_z']

    legs = ['L1', 'R1', 'L2', 'R2', 'L3', 'R3']
    
    for l in legs:
        lxs = ['%s' % l + i for i in xs]
        lys = ['%s' % l + i for i in ys]
        lzs = ['%s' % l + i for i in zs]
        xvals = df[lxs].values
        yvals = df[lys].values
        zvals = df[lzs].values
        for ix, i in enumerate(lxs):
            key = i.split('_')[0] + '_pt_position'
            vals = list(zip(xvals[:, ix], yvals[:, ix], zvals[:, ix]))
            df[key] = [np.array(i) for i in vals]
    return df

def get_closest_joint(val_array, pt_cols, joint):
    """
    Finds the closest joint to a given joint in a point cloud.
    Parameters:
    - val_array (numpy.ndarray): The array of feature vectors representing the point cloud.
    - pt_cols (list): The list of joint names corresponding to the feature vectors.
    - joint (str): The name of the joint to find the closest joint to.
    Returns:
    - closest (int): The index of the closest joint in the point cloud.
    - dist (float): The distance between the given joint and the closest joint.
    """
    
    kdt = KDTree(val_array, leaf_size=3, metric='euclidean')

    #Id of our joint of interest
    idx = [ix for ix, i in enumerate(pt_cols) if i == joint][0]
    X = val_array[idx].reshape(1,3) #corresponding feature vector for that joint

    #Query for the top 2 nearest neighbors since the first will be itself
    ds, similar_idxs = kdt.query(X, k=2)
    closest = similar_idxs[0][1]
    dist = ds[0][1]
    
    return closest, dist

def get_joint_distances(joint, df, pt_cols=[]):
    """
    Calculate the distances between a given joint and the closest joint in a DataFrame.
    Parameters:
    - joint (str): The joint to calculate distances from.
    - df (pandas.DataFrame): The DataFrame containing the joint positions.
    - pt_cols (list, optional): The columns in the DataFrame that contain the joint positions. 
                                If not provided, all columns containing 'pt_position' in their names will be used.
    Returns:
    - distances (tuple): A tuple of distances between the given joint and the closest joint for each row in the DataFrame.
    - closest (tuple): A tuple of the closest joint for each row in the DataFrame.
    """
    
    if len(pt_cols) == 0:
        pt_cols = [i for i in df.columns if 'pt_position' in i]
    vals = df.apply(lambda row: np.vstack(row[pt_cols].values), axis=1)
    df['vals'] = vals
    val_arrays = np.stack(df['vals'].values)
    closest_joints = df.apply(lambda row: get_closest_joint(row['vals'],pt_cols,joint), axis=1)
    closest, distances = zip(*closest_joints)

    return closest, distances

def interpolate_leg_vectors(df, suffix = ''):
    '''Interpolates vectors between consecutive joints of each leg to create a smooth representation.
        - df (pandas.DataFrame): The input DataFrame containing joint positions.
        - suffix (str, optional): A suffix to append to joint column names. Defaults to ''.
    Outputs:
        - dict: A dictionary where keys are leg prefixes (e.g., 'L1', 'R1') and values are numpy arrays
          representing interpolated vectors for each leg.'''
    
    prefixes = ['L1','L2','L3',
                'R1','R2','R3']

    leg_vector_dict = {}

    for prefix in prefixes:
        leg_points = [f'{prefix}{joint}_pt_position{suffix}' for joint in ['B', 'C', 'D', 'E']]
        leg_vectors = []

        for i in range(len(leg_points) - 1):
            start_point = np.vstack(df[leg_points[i]].values)
            end_point = np.vstack(df[leg_points[i + 1]].values)
            vector = np.linspace(start_point, end_point, num=100, axis=1)
            leg_vectors.append(vector)

        # Combine all vectors and include original joint positions
        leg_vectors = np.hstack(leg_vectors)
        leg_vector_dict[prefix] = leg_vectors
    return leg_vector_dict

def distance_between_legs(leg_1,leg_2,threshold, metric='euclidean'):
    """
    Computes the pairwise distance between two collections of 3d points to represent leg points.

    Args:
        leg_1 (numpy.ndarray): First collection of data points.
        leg_2 (numpy.ndarray): Second collection of data points.
        metric (str, optional): The distance metric to use. Defaults to 'euclidean'.

    Returns:
        numpy.ndarray: Pairwise distances between X and Y.
    """
    distances = cdist(leg_1, leg_2, metric='euclidean')
    

    idxs = np.argwhere(distances < threshold)
    filtered_idxs = [i for i in idxs if not (i[0] < 100 and i[1] < 100)]
    if filtered_idxs:
        filtered_array = np.vstack(filtered_idxs)
        mean_values = filtered_array.mean(axis=0)
        filtered_d = distances[filtered_array[:,0],filtered_array[:,1]]
        d = np.mean(filtered_d)
        leg_point_1 = mean_values[0]
        leg_point_2 = mean_values[1]
    else:
        d = np.min(distances)
        idx = np.unravel_index(np.argmin(distances, axis=None), distances.shape)
        leg_point_1 = idx[0]
        leg_point_2 = idx[1]                      
    return d, leg_point_1, leg_point_2


def classify_contacts(row, speed=2.0, d1=0.5):
    '''Classifies whether a contact event occurs based on distance and velocity thresholds.
    Inputs:
        - row (pandas.Series): A row of the DataFrame containing relevant data for classification.
        - distance (float, optional): The distance threshold for contact. Defaults to 0.13.
        - speed (float, optional): The velocity threshold for contact. Defaults to 1.0.
    Outputs:
    tuple: A tuple containing:
        - is_contact (bool): True if the row satisfies contact criteria, False otherwise.
        - pass_level (int): The level at which the classification failed (if any).'''
    d = row.distance
    threshold = row.threshold
    closest = row.other_leg_point
    root = row.root_leg_point
    root_leg = row.root_leg
    close_leg = row.other_leg

    root_velo_tatip = abs(row[f'{root_leg}_smoothed_velo'])
    close_velo_tatip = abs(row[f'{close_leg}_smoothed_velo'])

    root_velo_tita = abs(row[f'{root_leg}_D_smoothed_velo'])
    close_velo_tita = abs(row[f'{close_leg}_D_smoothed_velo'])

    root_velo_feti = abs(row[f'{root_leg}_C_smoothed_velo'])
    close_velo_feti = abs(row[f'{close_leg}_C_smoothed_velo'])

    root_d1_tatip = abs(row[f'{root_leg}_smoothed_velo_d1'])
    close_d1_tatip = abs(row[f'{close_leg}_smoothed_velo_d1'])

    root_d1_tita = abs(row[f'{root_leg}_D_smoothed_velo_d1'])
    close_d1_tita = abs(row[f'{close_leg}_D_smoothed_velo_d1'])

    root_d1_feti = abs(row[f'{root_leg}_C_smoothed_velo_d1'])
    close_d1_feti = abs(row[f'{close_leg}_C_smoothed_velo_d1'])


    pass_level = 0
    is_contact = False
    if d < threshold: #1
        if root < 100 and closest < 100: #2 cannot be body points that are close
            pass_level = 2
        else:
            if at_least_two_meet_threshold(root_velo_tatip, root_velo_tita, root_velo_feti, speed):
            #if root_smoothed_velo > speed or close_smoothed_velo > speed: #3
                is_contact = True
            elif at_least_two_meet_threshold(close_velo_tatip, close_velo_tita, close_velo_feti, speed):
                is_contact = True
            elif at_least_two_meet_threshold(root_d1_tatip, root_d1_tita, root_d1_feti, d1):
                is_contact = True
            elif at_least_two_meet_threshold(close_d1_tatip, close_d1_tita, close_d1_feti, d1):
                is_contact = True
            else:
                pass_level = 3
    else:
        pass_level = 1

    return is_contact, pass_level

def get_trial_df(df, suffix=''):   
    '''Processes a DataFrame to compute distances and classify contact events for each trial.
    Inputs:
        - df (pandas.DataFrame): The input DataFrame containing trial data.
        - threshold (float, optional): The distance threshold for contact classification. Defaults to 0.13.
        - suffix (str, optional): A suffix to append to joint column names. Defaults to ''.
    Outputs:
        - pandas.DataFrame: A DataFrame containing trial data with additional columns for distances,
                      contact classification, and other computed metrics.'''  
    
    threshold = df.threshold.iloc[0]
    print('threshold:', threshold)
    print(df.fullfile.iloc[0])
    leg_vectors = interpolate_leg_vectors(df, suffix=suffix)
    legs = ['L1','L2','L3',
            'R1','R2','R3']

    root_leg = 'L1'
    leg_df = pd.DataFrame()
    for leg in legs:
        if leg != root_leg:
            print(root_leg, leg)
            ds = []
            leg1_points = []
            leg2_points = []
            trial_length = leg_vectors['L1'].shape[0]
            for f in df.fnum.tolist(): 
                leg1 = leg_vectors[root_leg][f,:,:]
                leg2 = leg_vectors[leg][f,:,:]
                d, leg_point_1, leg_point_2 = distance_between_legs(leg1,leg2,threshold)
                ds.append(d)
                leg1_points.append(leg_point_1)
                leg2_points.append(leg_point_2)

            temp_df = pd.DataFrame()
            temp_df['fnum'] = df.fnum.tolist()
            temp_df['root_leg'] = [root_leg] * len(ds)
            temp_df['distance'] = ds
            temp_df['other_leg'] = [leg]*len(ds)
            temp_df['root_leg_point'] = leg1_points
            temp_df['other_leg_point'] = leg2_points

            joined_df = df.merge(temp_df, on='fnum', how='right',
                                suffixes=['_df','_leg'])

            contact_class = joined_df.apply(lambda row: classify_contacts(row), axis=1)
            contacts = [i[0] for i in contact_class]
            pass_level = [i[1] for i in contact_class]
            joined_df['is_contact'] = contacts
            joined_df['pass_level'] = pass_level   

            leg_df = pd.concat([leg_df,joined_df])
    #leg_df = leg_df.query('is_contact == True')
    return leg_df

def deduplicate_frames(df):
    '''Deduplicates frames in the DataFrame by keeping the best row for each frame based on contact status and distance.
    Inputs:
        - df (pandas.DataFrame): The input DataFrame containing frame data.
    Outputs:
        - pandas.DataFrame: A deduplicated DataFrame with one row per frame.'''

    # Sort the dataframe based on the conditions
    df_sorted = df.sort_values(by=['fnum', 'is_contact', 'distance'], ascending=[True, False, True])

    # Drop duplicates based on 'fnum', keeping the first occurrence (which will be the best row based on the sorting)
    df_deduplicated = df_sorted.drop_duplicates(subset='fnum', keep='first')

    # Reset the index of the resulting dataframe
    df_deduplicated = df_deduplicated.reset_index(drop=True)

    return df_deduplicated

def get_sweep(df, min_bout_length=4):
    '''Identifies and labels sweeps (bouts) of contact events in the DataFrame.
    Inputs:
    - df (pandas.DataFrame): The input DataFrame containing contact data.
    - min_bout_length (int, optional): The minimum length of a bout to be considered valid. Defaults to 4.
    Outputs:
    pandas.DataFrame: The DataFrame with an additional column 'sweep_number' indicating the sweep number
                      for each row. Rows not part of a valid sweep are labeled with 0.'''
    # Initialize the bout number and a flag to indicate if we are in a bout
    sweep_number = 0
    in_bout = False

    # Create a new column for bout numbers
    df['sweep_number'] = 0

    # Iterate through the dataframe
    for i in range(len(df)):
        if df.loc[i, 'is_contact']:
            if not in_bout:
                # Start of a new bout
                sweep_number += 1
                in_bout = True
            # Assign the current bout number
            df.loc[i, 'sweep_number'] = sweep_number
        else:
            # End of a bout
            in_bout = False

    # Calculate bout lengths
    bout_lengths = df[df['sweep_number'] > 0].groupby('sweep_number').size()

    # Filter out bouts with length less than or equal to min_bout_length
    valid_bouts = bout_lengths[bout_lengths >= min_bout_length].index

    # Update bout numbers in the dataframe to 0 for invalid bouts
    df.loc[~df['sweep_number'].isin(valid_bouts), 'sweep_number'] = 0

    inter_df, length_dict = get_inter_df(df)

    df = get_recombined_sweeps(df, inter_df, length_dict, min_bout_length=min_bout_length)

    return df

def get_inter_df(df):
    '''Tracks all the unique bouts (transitions between states of sweep or not sweep) and the
    length of each of these bouts is stored in a dicitonary. The original sweep number is 
    stored in the inter_df dataframe along with it's unique bout_number.'''
    val = 0
    count = 1
    counts = []
    vals = [0]
    sweep_numbers = df['sweep_number'].values
    for ix,i in enumerate(sweep_numbers):
        if ix > 0:
            prev = sweep_numbers[ix - 1]
            if i == prev:
                count += 1
                vals.append(val)
            else:
                counts.append(count)
                count = 1
                val +=1
                vals.append(val)
    counts.append(count)

    unique_vals = list(set(vals))
    length_dict = {}
    for i in range(len(unique_vals)):
        length_dict[unique_vals[i]] = counts[i]


    inter_df = pd.DataFrame()
    inter_df['sweep_number'] = sweep_numbers
    inter_df['bout_number'] = vals
    return inter_df, length_dict

def get_recombined_sweeps(df,inter_df, length_dict, min_bout_length = 4):
    '''To combine sweeps that are separated by short gaps that are less than the threshold.'''
    prev_sweep = inter_df.sweep_number.iloc[0]
    new_sweeps = []
    for ix, i in inter_df.iterrows():
        sweep = i.sweep_number
        bout = i.bout_number

        length = length_dict[bout]
        if length < min_bout_length:
            new_sweeps.append(prev_sweep)
        else:
            new_sweeps.append(sweep)
            prev_sweep = sweep
    #print(np.array(new_sweeps))

    count = 1
    prev_sweep = new_sweeps[0]
    combined_sweeps = [new_sweeps[0]]
    for s in new_sweeps[1:]:
        if s != 0:
            if prev_sweep == 0:
                count += 1
            combined_sweeps.append(count)
        else:
            combined_sweeps.append(0)
        prev_sweep = s
    np.array(combined_sweeps)

    df['sweep_number_new'] = combined_sweeps
    return df


def process_fullfile(df,fullfile):
    '''Processes a subset of the DataFrame corresponding to a specific file, 
    identifying contact events and sweeps.
    Inputs:
        - df (pandas.DataFrame): The input DataFrame containing all data.
        - fullfile (str): The specific file identifier to filter the DataFrame.
        - threshold (float, optional): The distance threshold for contact classification. Defaults to 0.13.
    Outputs:
        - pandas.DataFrame: A DataFrame containing processed data for the specified file, including contact
                      events and sweep information.'''
    
    subset = df[df['fullfile'] == fullfile]
    fullfile_df = get_trial_df(subset)
    deduplicated = deduplicate_frames(fullfile_df)
    bout_df = get_sweep(deduplicated)
    return bout_df

def get_contact_dist(fullfile, fnum, prefixes=['L1']):
    subset = complete_pts.query('fullfile == @fullfile & fnum == @fnum')
    pt = subset['contact_pt'].values[0]  # Get the contact point for the specified sample and frame
    # if pt.shape[0] > 1:
    #     pt = pt[:1]
    pt = np.array(pt).reshape(1,3)  # Ensure pt is a numpy array
    suffix = ''
    leg_vector_dict = {}

    for prefix in prefixes:
        leg_points = [f'{prefix}{joint}_pt_position{suffix}' for joint in ['A','B', 'C', 'D', 'E']]
        leg_vectors = []

        for i in range(len(leg_points) - 1):
            start_point = np.vstack(subset[leg_points[i]].values)
            end_point = np.vstack(subset[leg_points[i + 1]].values)
            vector = np.linspace(start_point, end_point, num=100, axis=1)
            leg_vectors.append(vector)

        # Combine all vectors and include original joint positions
        leg_vectors = np.hstack(leg_vectors)
        leg_vector_dict[prefix] = leg_vectors

    leg = prefixes[0]  # Assuming we are only interested in the first leg
    leg_1 = leg_vector_dict[leg][0, :, :]  # Extract the first leg's vectors
    #print(leg_1.shape, pt.shape,subset.blind_id.values[0])
    # Calculate the closest point on the vector to the contact point
    distances = cdist(leg_1, pt)  # Compute pairwise distances between leg_1 points and pt
    closest_index = np.argmin(distances)  # Find the index of the closest point
    #print(closest_index, distances.shape)
    dist = np.min(distances)  # Get the distance to the closest point
    closest_point = leg_1[closest_index,:]  # Retrieve the closest point
    return closest_index, dist, closest_point
if __name__ == "__main__":
    # Example usage or test code can be added here
    pass