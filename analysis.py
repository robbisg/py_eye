import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as sp

def mean_analysis(data, trial_info, downsampling=False, **kwargs):
    #Da rivedere
    #print data
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
            #print field
            #print c
            #print data[field][mask]
            #print np.unique(data['Trial'][mask])
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

def group_function(s_data, fields, functor=np.mean):
    
    r = dict()
    
    for field in fields:
        
        r[field] = functor(s_data[field], axis=1)
    
    return r
    

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

def split_data(d_data, fields, chunk_time=0.02, functor=group_function):
    
    '''
    Returns splitted data
    '''
    
    
    chunk_points = np.rint(chunk_time * d_data['SampleRate'])
    
    data = d_data['data']
    splitted_data = d_data.copy()
    
    t_mask = False
    for trial in np.unique(data['Trial']):
        
        m_trial = data['Trial'] == trial
        
        d_trial = data[m_trial]
        n_chunks = np.floor(len(d_trial)/chunk_points)
        
        included_points = n_chunks * chunk_points
        trial_first = np.nonzero(m_trial)[0][0]
        
        m_trial[trial_first+n_chunks:] = False
        t_mask = t_mask + m_trial
        
        d_trial = d_trial[:included_points]
        
        d_trial = np.array(np.split(d_trial, n_chunks))
        f_trial = functor(d_trial, fields)       
        
        for field in fields:
            splitted_data['data'][field][m_trial] = f_trial[field]
    
    
    return splitted_data['data'][np.array(t_mask, np.bool)], 1/chunk_time

def analyze_timecourse(data, trial_cond, sample_rate, **kwargs):
    
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
    
    results = build_result_structure(fields, conditions)
    
    f = plt.figure()
    
    for condition in conditions:
        i = 0
        for field in fields:
            i = i + 1
            
            a = f.add_subplot(len(fields),1,i)
            
            m_cond_trial = trial_cond[column] == condition
            cond_trial = trial_cond[m_cond_trial]
            data_list = []
            dim = np.array([])
            for trial in cond_trial['Trial']:
                d_list = data[data['Trial'] == trial][field]
                
                if d_list.shape[0] != 0:
                    dim = np.append(dim, d_list.shape[0])
                else:
                    continue
                
                dim = np.append(dim, d_list.shape[0])
                data_list.append(d_list)
                
            min_ = np.min(dim)
            
            data_list = [d[:min_] for d in data_list]

            data_list = np.vstack(data_list)
            mean = np.mean(data_list, axis=0)
            std = np.std(data_list, axis=0)
            
            results[field][condition]['mean'] = mean
            results[field][condition]['std'] = std
            '''
            xx = np.linspace(0, len(mean), len(mean))
            yy = mean
            
            smooth = sp.UnivariateSpline(xx, yy, s=1)
            y_smooth = smooth(xx)
            
            a.plot(y_smooth, alpha=0.5)
            '''
            a.plot(mean)
            a.set_title(field)
            a.legend(conditions)
            xticks = np.arange(0, min_/sample_rate, np.around((min_/sample_rate)/7., decimals=2))
            #a.set_xlim((0, min/sample_rate))
            a.set_xticklabels(xticks)
            
    
    
            
    return results
    
    




"""
def mean_time(d_data, trial_info, time_chunks=20, **kwargs):
    
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
    
    data = d_data['data']
    
    for condition in conditions:
    
    return 0
"""
