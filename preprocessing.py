import numpy as np
import scipy.interpolate as sp
from scipy import arange, array, exp
import sys

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
        mean_mask = mean_outlier(data, std_thr, field)
        
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
    
def mean_outlier(data, std_thr, field):
       
    t_mask = np.array([])
    
    for trial in np.unique(data['Trial']):
        
        mask = data['Trial'] == trial
        m_data = data[mask]
        
        mean = np.mean(data[mask][field])
        
        mask_m = size_outlier(data=data[mask], 
                              max=mean+std_thr, 
                              min=mean-std_thr, 
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
 
 
def baseline_correction(data, mask, trial_info, fields, condition, type='previous'):
    
    
    if type == 'previous':
        c_data = remove_baseline_previous(data, mask, trial_info, fields, condition)
    else:
        foo()
      
    return c_data
    
def remove_baseline_previous(data, mask, trial_info, fields, condition):
    
    c_data = data.copy()
    
    fix = trial_info[trial_info['Condition'] == condition]
    
    
    
    for tr in fix:   
        for field in fields:
            trial = np.int(np.float(tr[0]))
            
            mask_trial = data['Trial'] == trial
            mask_baseline = mask_trial * mask
            
            if np.count_nonzero(mask_baseline) == 0:
                mask_baseline = data['Trial'] == trial + 1
                ext_baseline = data[field][mask_baseline]
                mean = np.mean(ext_baseline[:240])
            else:
                mean = np.mean(data[field][mask_baseline])
                
            out = 'Trial corrected: ' + str(trial+1)
            mask_condition = data['Trial'] == trial + 1
            
            sys.stdout.write('\r')
            sys.stdout.write(out)
            sys.stdout.flush()
            
            c_data[field][mask_condition] = data[field][mask_condition] - mean
            
    print '\n'
    return c_data    
        
def interpolate_trial(data, trial_info, fields, valid_mask):
    
    i = 0
    j = 0
    for t in trial_info['Trial']:
        
        j = j + 1
        
        trial = np.int(np.float(t))
        mask_trial = data['Trial'] == trial
        
        outlier_length = np.count_nonzero(1 - valid_mask[mask_trial])
        trial_length = len(data['Trial'][mask_trial])
        
        if (trial_length)*0.5 < outlier_length:
            print 'Bad Trial'
            i = i + 1
            valid_mask[mask_trial] = False
        else:
            try:
                data[mask_trial] = interpolate(data, valid_mask, mask_trial, fields)
                valid_mask[mask_trial] = True
            except ValueError, err:
                continue
        perc = 0    
        perc = np.float(j)/np.float(len(trial_info['Trial']))
        
        output = str(perc * 100.)+' % completed'
        sys.stdout.write('\r')
        sys.stdout.write(output)
        sys.stdout.flush()
        #print str(perc*100.),"  % completed         \r",  
    
    print '\nBad Trials: '+str(i)+' percentage: '+str(i/trial*0.5)
    
    return [data, valid_mask]

def interpolate(data, valid_mask, mask_trial, fields):
    
    mask = valid_mask[mask_trial]
    m_data = data[mask_trial]
    
    #print 'Interpolating trial: '+str(np.unique(m_data['Trial']))
    
    for field in fields:
        
        xx = np.linspace(0, len(mask), len(mask))
        
        x = xx[mask]
        f = m_data[field][mask]
        
        f_inter = sp.interp1d(x, f, kind='cubic')
        
        f_extra = extrap1d(f_inter)
        
        pt = f_extra(xx)      
        
        m_data[field] = np.array(pt, dtype=np.float32)
        
    

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
       
if __name__== '__main__':
    import sys
    
       