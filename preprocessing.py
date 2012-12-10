import numpy as np
import scipy.interpolate as sp
from scipy import arange, array, exp
import sys
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import Ridge

def pixel_to_mm (d_data, fields, factor):
    
    data = d_data['data']
    
    for f in fields:
        data[f] = data[f]/factor
    
    d_data['data'] = data
    
    return d_data


def mm_to_pixel (d_data, fields, factor):
    
    data = d_data['data']
    
    for f in fields:
        data[f] = data[f]*factor
    
    d_data['data'] = data
    
    return d_data

def remove_outliers(d_data, **kwargs):
    
    for arg in kwargs:
        if arg == 'max_pupil_size':
            max = np.float(kwargs[arg])
        if arg == 'min_pupil_size':
            min = np.float(kwargs[arg])
        if arg == 'std_thr':
            std_thr = np.float(kwargs[arg])        
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')
        if arg == 'window':
            window = np.float(kwargs[arg])            
        if arg == 'pupil_factor':
            factor = np.float(kwargs[arg])
    
    d_data = pixel_to_mm(d_data, fields, factor)
        
    data = d_data['data']
    
    outlier_mask = False
    for field in fields:
        size_mask = size_outlier(data, max, min, field)
        mean_mask = mean_outlier(data, std_thr, field, size_mask)
        
        outlier_mask = outlier_mask + mean_mask + size_mask
        
    valid_mask = window_outlier(outlier_mask, window)
    valid_mask = True - valid_mask
    
    #data = data[final_mask]
    
    #d_data['data'] = data
    
    return np.array(valid_mask, dtype=np.bool)
    

def size_outlier(data, max, min, field):
    
      
    mask_min = data[field] < min
    mask_max = data[field] > max   
    
    total = mask_min + mask_max
        
    return total
    
def mean_outlier(data, std_thr, field, p_mask):
       
    t_mask = np.array([])
    
    for trial in np.unique(data['Trial']):
        
        mask = data['Trial'] == trial
              
        mask_trial_masked = ~p_mask[mask]

        mean = np.mean(data[field][mask][mask_trial_masked])
        std = np.std(data[field][mask][mask_trial_masked])

        mask_m = size_outlier(data=data[mask], 
                              max=mean+(std*std_thr), 
                              min=mean-(std*std_thr), 
                              field=field)
        
        t_mask = np.hstack((t_mask, mask_m))
    
    return t_mask
        
    
def window_outlier(mask, window):
    
    for index in np.nonzero(np.int_(mask))[0]:
        begin = index - window
        end = index + window
        
        if begin < 0:
            begin = 0
        if end > len(mask) - 1:
            end = len(mask) - 1
        
        mask[begin:end] = True
        
    return mask
 
 
def baseline_correction(data, valid_mask, trial_info, fields, condition, type='previous'):
    
    
    if type == 'previous':
        c_data = remove_baseline_previous(data, valid_mask, trial_info, fields, condition)
    else:
        foo()
      
    return c_data
    
def remove_baseline_previous(data, valid_mask, trial_info, fields, condition):
    
    #c_data = data.copy()
    
    fix = trial_info[trial_info['Condition'] == condition]

    for tr in fix:   
        for field in fields:
            trial = np.int(np.float(tr[0]))
            
            mask_trial = data['Trial'] == trial
            mask_baseline = mask_trial * valid_mask
            
            if np.count_nonzero(mask_baseline) == 0:
                mask_trial = data['Trial'] == trial + 1
                mask_baseline = mask_trial * valid_mask
                if np.count_nonzero(mask_baseline) != 0:
                    ext_baseline = data[field][mask_baseline]
                    mean = np.mean(ext_baseline[:240])
                else:
                    mean = 0
            else:
                mean = np.mean(data[field][mask_baseline])
                
            out = 'Trial corrected: ' + str(trial+1)
            mask_condition = data['Trial'] == trial + 1
            
            mask_condition_masked = mask_condition * valid_mask
            if np.count_nonzero(mask_condition_masked) == 0:
                continue
                
            sys.stdout.write('\r')
            sys.stdout.write(out)
            sys.stdout.flush()
            
            data[field][mask_condition_masked] = data[field][mask_condition_masked] - mean
            
    print '\n'
    return data    
        
def interpolate_trial(data, trial_info, fields, valid_mask):
    
    i = 0
    j = 0
    
    bad_trials = []
    for t in trial_info['Trial']:
        
        j = j + 1
        
        trial = np.int(np.float(t))
        mask_trial = data['Trial'] == trial
        
        outlier_length = np.count_nonzero(~valid_mask[mask_trial])
        trial_length = np.count_nonzero(mask_trial)
        
        if (trial_length)*0.5 < outlier_length:
            bad_trials.append(trial)
            i = i + 1
            valid_mask[mask_trial] = False
        else:
            try:
                #data[mask_trial] = sklearn_fit(data, valid_mask, mask_trial, fields)
                data[mask_trial] = interpolate(data, valid_mask, mask_trial, fields)
                valid_mask[mask_trial] = True
            except ValueError, err:
                valid_mask[mask_trial] = False
                continue
        perc = 0    
        perc = np.float(j)/np.float(len(trial_info['Trial']))
        
        output = str(perc * 100.)+' % completed'
        sys.stdout.write('\r')
        sys.stdout.write(output)
        sys.stdout.flush()
        #print str(perc*100.),"  % completed         \r",  
    
    print '\nBad Trials: '+str(i)+' percentage: '+str(np.float(i)/np.float(len(trial_info)))+'\n'
    print bad_trials
    
    return [data, valid_mask]

def interpolate(data, valid_mask, mask_trial, fields):
    
    mask = valid_mask[mask_trial]
    m_data = data[mask_trial]
    
    #print 'Interpolating trial: '+str(np.unique(m_data['Trial']))
    ridge = Ridge()
    
    for field in fields:
        
        xx = np.linspace(0, len(mask), len(mask))
        
        x = xx[mask]
        y = m_data[field][mask]
        
        f_inter = sp.interp1d(x, y, kind='linear')
        
        f_extra = extrap1d(f_inter)
        
        yy = f_extra(xx)
        
        smooth = sp.UnivariateSpline(xx, yy, s=1)
        y_smooth = smooth(xx)
        
        try:
            ridge.fit(np.vander(xx, 11), y_smooth)
            y_fit = ridge.predict(np.vander(xx, 11))
        except LinAlgError,err:
            ridge.fit(np.vander(xx, 9), y_smooth)
            y_fit = ridge.predict(np.vander(xx, 11))
            
        m_data[field] = np.array(y_fit, dtype=np.float32)

    return m_data

def foo():
    return 'foo'



def extrap1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
    
        
        if x < xs[0]:
            return ys[0]+(x-xs[0])*(ys[1]-ys[0])/(xs[1]-xs[0])
        elif x > xs[-1]:
            return ys[-1]+(x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike   

    
def correct_mask(data, valid_mask, fields):
    
    for trial in np.unique(data['Trial']):
        mask_trial = data['Trial'] == trial
        
        valid_masked = valid_mask[mask_trial]
        
        if valid_masked[-1] == False:
            index = np.nonzero(mask_trial)[0][-1]
            valid_mask[index] = True
            for field in fields:
                data[field][index] = 0.
        
        if valid_masked[0] == False:
            index = np.nonzero(mask_trial)[0][0]
            valid_mask[index] = True
            for field in fields:
                data[field][index] = 0. 
                
    return valid_mask