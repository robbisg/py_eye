import os
import numpy as np

def mean_analysis(data, trial_info, downsampling=False, **kwargs):
    #Da rivedere
    print data
    for arg in kwargs:
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')
        if arg == 'conditions':
            try:
                conditions = np.int_(kwargs[arg].split(','))
            except ValueError, err:
                conditions = kwargs[arg].split(',')
                continue
        if arg == 'behavioural_field':
            column = kwargs[arg]
    
    c_mask = []
    for condition in conditions:
        m_cond_trial = trial_info[column] == condition
        
        mask_cond_data = build_mask(data, trial_info[m_cond_trial])
        c_mask.append(mask_cond_data)
    
    
    results = build_result_structure(fields, conditions)
    
    for field in fields:
        for c, mask in zip(conditions, c_mask):
            print field
            print c
            print data[field][mask]
            print np.unique(data['Trial'][mask])
            results[field][c]['mean'] = np.mean(data[field][mask])     
            results[field][c]['std']  = np.std(data[field][mask])
    
    print results
    
    return results

def build_mask(data, trial_info, points=None):
    """
    Build a mask based on trials contained in trial_info parameter
    """
    total_mask = False
    
    for t in trial_info['Trial']:
        tr = np.float(t)
        mask = data['Trial'] == tr
        total_mask = total_mask + mask
        
    return total_mask

def group_mean():
    
    return None
    

def build_result_structure(conditions, fields):
    
    r = dict()
    
    for cond in conditions:
        r[cond] = dict()
        for field in fields:
            r[cond][field] = dict()
            
    return r


def open_behavioural(path, subj):
    
    import xlrd
    fn = os.path.join(path, subj)
    
    book = xlrd.open_workbook(fn)
    sh = book.sheet_by_index(0)
    
    behavioural = np.array(zip(
                               sh.col_values(9)[1:], 
                     np.float_(sh.col_values(17)[1:]),
                       np.int_(sh.col_values(8)[1:])
                            ), 
                        dtype=[('Condition', np.str_,4),
                               ('Accuracy', np.int_, 1),
                               ('Combination', np.int_, 1),])
    
    return behavioural

def mean_time(d_data, trial_info):
    
    data = d_data['data']
    
    
    
    
    return 0