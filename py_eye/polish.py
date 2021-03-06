#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################

import numpy as np

def clear_eyeblink(d_data):
    
    data = d_data['data']
    mask_eb = data['L_Raw_X'] != 0 #Good values
    
    t_info = extract_trials_info(d_data)
    
    i = 0
    d_trial = []
    for t in t_info:
        
        mask_tr = data['Trial'] == t[0]
        mask = mask_eb[mask_tr]
        
        eb = np.count_nonzero(1-mask)

        if eb > np.count_nonzero(mask_tr) * 0.5:
            mask_eb[mask_tr] = False
            d_trial.append(t[0])
            i = i + 1
    
    data = data[mask_eb]
    
    d_data['data'] = data
    
    
    print 'Trials deleted: '+str(i)+' deleted trials: '+str(d_trial)
    
    return d_data


def clear_trials(d_data, trial_list):
    
    data = d_data['data']
    
    mask = data.T['Trial'] != trial_list[0]
    for trial in trial_list[1:]:
        mask =  np.logical_and(mask, data.T['Trial'] != trial) #First trial is unuseful
    
    data = data[mask]
    
    d_data['data'] = data
    
    return d_data

def detect_errors(d_data, trial_info, trial_expected, sensibility=0.2):
    '''
    trial_expected: numpy array with trial timing
                    example: if the experiment contains two condition the first 
                    with time 1 second and the latter with 3 seconds
                    trial_expected = np.array([1.,3.])
    sensibility : error in registration so a trial could last instead of 3 seconds
                    3 + sensibility
    '''    
    
    max_time = trial_expected.max() + sensibility
    max_points = max_time * d_data['SampleRate']
    errors = []
    
    trials = extract_trials_info(d_data)
    #print max_time
    for trial in trials:
        if trial[1] > max_points:
            errors.append(trial) 
            
       
    return np.array(errors)


def detect_pause(trial_info):
    
    
    return
            
def correct_data(d_data, trial_info, errors, trial_expected):
    '''
    trial_expected: numpy array with trial timing
                    example: if the experiment contains two condition the first 
                    with time 1 second and the latter with 3 seconds
                    trial_expected = np.array([1.,3.])
    
    '''
    
    
    data = d_data['data']
    
    # timing = trial_expected['trial_timing']
    
    i = 0
    while len(errors) > 0:
        
        i = i + 1
        error = errors[0]
        trial_damaged = error[0]
        mask = data['Trial'] == trial_damaged
        
        first_err_index = mask.nonzero()[0][0]       
        
        m_error = trial_info['Trial'] == error[0]#mask
        i_error = m_error.nonzero()[0][0]#first index in data of the error
        
        
        i_error_pre = i_error-2
        if i_error_pre < 0:
            i_error_pre = i_error + 1
        
        
        new_duration = trial_info[i_error_pre][1]
        #print new_duration
        #aggiungi 1 a tutti i trial da first_err_index + new_duration
        data['Trial'][first_err_index+new_duration:] = data['Trial'][first_err_index+new_duration:]+1
        
        if (error[1] > (trial_expected.sum() * d_data['SampleRate'])):
            mask = data['Trial'] != trial_damaged + 1
            data = data[mask]
            mask_new = data['Trial'] > trial_damaged + 1
            data['Trial'][mask_new] = data['Trial'][mask_new] - 1
        
        d_data['data'] = data
        
        trial_info = extract_trials_info(d_data)
        errors = detect_errors(d_data, trial_info, trial_expected)
        
        print errors
    
     
    return d_data  
        
    
        
def extract_trials_info(d_data):
    
    data = d_data['data']
    
    trial_info = []
    
    for trial in np.unique(data['Trial']):
        mask_trial = data['Trial'] == trial
        trial_d = data[mask_trial]
        #  print str(trial)+' '+str(len(trial_d.T[0]))+ \
        #  ' '+str(trial_d.T[0][len(trial_d.T[0])-1] - trial_d.T[0][0] )
        
        trial_info.append((trial, len(trial_d['Time']), \
                           trial_d['Time'][len(trial_d['Time'])-1] - trial_d['Time'][0]))
    
    dt = np.dtype([('Trial', np.int16) , ('Length', np.int32), ('Index', np.int32)])
       
    return np.array(trial_info, dtype=dt)
      


def correct_fixation(d_data, trial_info, fix_time):
    
    # Get fixation trial info
    trial_fix = trial_info[::2]
    # Get data
    data = d_data['data']
    # Number of points for fixation trial
    fix_points = fix_time * d_data['SampleRate']
    
    for trial in np.unique(trial_fix['Trial']):
       
        # Get mask for fix and trial
        mask_fix = data['Trial'] == trial
        mask_trial = data['Trial'] == (trial+1) 
        
        # Compare true fix_point with established fix points
        if np.count_nonzero(mask_fix) < fix_points:
            
            # Difference of points
            points = fix_points - np.count_nonzero(mask_fix)
            
            # Build a vector with the trial size
            subvec = np.ones_like(np.nonzero(mask_trial)[0]) * trial+1
            
            # Replace first points with fixation trial number
            subvec[:int(points)] = trial
            # Replace the new trial vector
            data['Trial'][mask_trial] = subvec
            
            
    d_data['data'] = data
    
    return d_data
        