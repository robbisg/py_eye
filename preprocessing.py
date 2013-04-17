import numpy as np
import scipy.interpolate as sp
from scipy import arange, array, exp
import sys
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcess
from sklearn.linear_model import Ridge
from sklearn import mixture
from sklearn.cluster import KMeans

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

def merge_fields(d_data, **kwargs):
    
    
    data = d_data['data']
    
    for arg in kwargs:    
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')
    
    i = 0
    for field in fields:
        if i == 0:
            f = field
            stacked = data[field]
        else:
            
            stacked = np.vstack((stacked, data[field]))
        i = i + 1    
    
    data[f] = np.mean(stacked, axis=0)
    
    d_data['data'] = data
    
    return d_data, f            

def remove_outlier_mshift(d_data, **kwargs):
    
    for arg in kwargs:
      
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')
        if arg == 'window':
            window = np.float(kwargs[arg])            
        if arg == 'pupil_factor':
            factor = np.float(kwargs[arg])
    
    from sklearn.cluster import MeanShift, estimate_bandwidth
        
    data = d_data['data']
        
    outlier_list = dict()
    total_outlier = False
    
    
    for field in fields:
        
        outlier_list[field] = []
        
        for trial in np.unique(data['Trial']):
            indexes = []
            f = plt.figure()
            a = f.add_subplot(111)
            mask_trial = data['Trial'] == trial
            masked_data = data[mask_trial][field]
            
            X = masked_data[np.newaxis].T
            
            bandwidth = estimate_bandwidth(X, quantile=0.2)
            if bandwidth < 0.9:
                bandwidth = 1
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            
            ms.fit(X)
            labels = ms.labels_
            
            '''
            i_outliers = np.argsort(clf.means_.squeeze())
            
            mean_top = clf.means_.squeeze()[i_outliers[2]]
            mean_low = clf.means_.squeeze()[i_outliers[0]]
            mean_true = clf.means_.squeeze()[i_outliers[1]]
            
            if mean_low == 0:
                indexes.append(i_outliers[0])
            if (mean_top - mean_true) > 3 * clf.covars_.squeeze()[i_outliers[1]]:
                indexes.append(i_outliers[2])
            if (mean_true - mean_low) > 3 * clf.covars_.squeeze()[i_outliers[1]]:
                indexes.append(i_outliers[0])
                
            outlier = np.zeros(np.shape(masked_data), np.bool)
            for i in indexes:
                outlier = outlier + (labels == i)
            
            
            outlier_list[field].append(outlier)
            
            out = 'Trial analyzed: '+str(trial+1)
            
            sys.stdout.write('\r')
            sys.stdout.write(out)
            sys.stdout.flush()
            '''
            a.plot(masked_data)
            a.plot(labels)
            #a.plot(outlier)
            
        outlier_list[field] = np.hstack(outlier_list[field])

        total_outlier = total_outlier + outlier_list[field]
        
    valid_mask = window_outlier(total_outlier, window)
    valid_mask = True - valid_mask
    
    return valid_mask
            
def remove_outliers_gmm(d_data, **kwargs):
    
    for arg in kwargs:
      
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')
        if arg == 'window':
            window = np.float(kwargs[arg])            
        if arg == 'pupil_factor':
            factor = np.float(kwargs[arg])
    
        
    data = d_data['data']
    
    clf = mixture.GMM(n_components=3, covariance_type='full')
    
    outlier_list = dict()
    total_outlier = False
    
    fb = plt.figure()
    for field in fields:
        
        outlier_list[field] = []
        b = fb.add_subplot(111)
        for trial in np.unique(data['Trial']):#[::7]:
            indexes = []
            #f = plt.figure()
            
            #a = f.add_subplot(111)
            #a.set_title(str(trial))
            mask_trial = data['Trial'] == trial
            masked_data = data[mask_trial][field]
            """
            if np.var(masked_data) < 0.1:
                n = 2
            else:
                n = 3            
            """
            n = 3
            
            #z_data = stats.zscore(masked_data)
             
            clf = mixture.GMM(n_components=n, covariance_type='full')
            
            labels = clf.fit(masked_data).predict(masked_data)
                        
            i_outliers = np.argsort(clf.means_.squeeze())
            
            mean_top = clf.means_.squeeze()[i_outliers[2]]
            mean_low = clf.means_.squeeze()[i_outliers[0]]
            mean_true = clf.means_.squeeze()[i_outliers[1]]
            
            if mean_low < -1:
                indexes.append(i_outliers[0])
            if (mean_top - mean_true) > 2 * clf.covars_.squeeze()[i_outliers[1]]:
                indexes.append(i_outliers[2])
            if (mean_true - mean_low) > 2 * clf.covars_.squeeze()[i_outliers[1]]:
                indexes.append(i_outliers[0])
                
            outlier = np.zeros(np.shape(masked_data), np.bool)
            for i in indexes:
                outlier = outlier + (labels == i)
            
            
            outlier_list[field].append(outlier)
            
            out = 'Trial analyzed: '+str(trial+1)
            
            print trial
            print np.mean(masked_data)
            print np.var(masked_data)
            print clf.means_.squeeze()
            print clf.covars_.squeeze()
            '''
            sys.stdout.write('\r')
            sys.stdout.write(out)
            sys.stdout.flush()
            '''
            b.scatter(np.mean(masked_data), np.var(masked_data))
            
            '''
            a.plot(masked_data)
            a.plot(np.linspace(np.mean(masked_data), np.mean(masked_data), len(masked_data)))
            a.plot(outlier)
            '''
        outlier_list[field] = np.hstack(outlier_list[field])

        total_outlier = total_outlier + outlier_list[field]
        
    valid_mask = window_outlier(total_outlier, window)
    valid_mask = True - valid_mask
    
    return valid_mask
        
def find_index_outlier(labels, means, covars):
    
    indexes = []
    
    i_outliers = np.argsort(means)
    
    if covars[i_outliers[1]] > 0.05:
        if means[i_outliers[0]] == 0.:
            true_index = i_outliers[2]
    
    
    
    mean_top = means[i_outliers[2]]
    mean_low = means[i_outliers[0]]
    mean_true = means[i_outliers[1]]
            
    if mean_low == 0:
        indexes.append(i_outliers[0])
    elif (mean_top - mean_true) > 2 * clf.covars_.squeeze()[i_outliers[1]]:
        indexes.append(i_outliers[2])
    elif (mean_true - mean_low) > 2 * clf.covars_.squeeze()[i_outliers[1]]:
        indexes.append(i_outliers[0])
    
   
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
    
    
        
    data = d_data['data']
    
    outlier_mask = False
    for field in fields:
        
        
        size_mask = size_outlier(data, max, min, field)
        #size_mask = window_outlier(size_mask, window)
        
        mean_mask = mean_outlier(data, std_thr, field, size_mask)
        #mean_mask = window_outlier(mean_mask, window)
        
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
    '''
    Outliers detected were increased using a window
    for each outlier a window of @window point will be build
    the outlier found is the center of the interval of +window
    and -window points
    '''
    
    for index in np.nonzero(np.int_(mask))[0]:
        begin = index - window
        end = index + window
        
        if begin < 0:
            begin = 0
        if end > len(mask) - 1:
            end = len(mask) - 1
        
        #is setted as true because the mask is true where we have
        #an outlier
        mask[begin:end] = True 
        
    return mask
 
 
def baseline_correction(data, valid_mask, trial_info, type='previous', **kwargs):
    
    points = 240
    
    for arg in kwargs:    
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')          
        if arg == 'baseline':
            condition = kwargs[arg]
        if arg == 'baseline_size':
            points = int(kwargs[arg])
    
    if type == 'previous':
        c_data = remove_baseline_previous(data, valid_mask, trial_info, 
                                          fields, points, condition)
    else:
        c_data = remove_baseline_trial(data, valid_mask, trial_info, 
                                       fields, points, condition)
      
    return c_data

def remove_baseline_trial(data, valid_mask, trial_info, fields, points, condition):
    
    for tr in trial_info:
        for field in fields:
            trial = np.int(np.float(tr[0]))
            
            mask_trial = data['Trial'] == trial
            mask_condition = mask_trial * valid_mask
            data_baseline = data[field][mask_condition][:points]          
            
            mean = np.mean(data_baseline)
            
            if np.isnan(mean):
                continue
                mean = 0
            
            mask_data = mask_trial * valid_mask
            
            if np.count_nonzero(mask_data) == 0:
                continue
                        
            data[field][mask_data] = data[field][mask_data] - mean
            
    return data

def remove_baseline_previous(data, valid_mask, trial_info, fields, points, condition):
    
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
                    mean = np.mean(ext_baseline[points:])
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
    
    print '\nBad Trials: '+str(i)+' percentage: '+str(np.float(i * 100)/np.float(len(trial_info)))+'\n'
    print bad_trials
    
    return [data, valid_mask]

def interpolate(data, valid_mask, mask_trial, fields):
    
    mask = valid_mask[mask_trial]
    m_data = data[mask_trial]
    
    #print 'Interpolating trial: '+str(np.unique(m_data['Trial']))
    
    for field in fields:
        
        xx = np.linspace(0, len(mask), len(mask))
        
        x = xx[mask]
        y = m_data[field][mask]
        
        f_inter = sp.interp1d(x, y, kind='linear')
        
        f_extra = extrap1d(f_inter)
        
        yy = f_extra(xx)
        
        smooth = sp.UnivariateSpline(xx, yy, s=5)
        y_smooth = smooth(xx)
        """
        try:
            ridge.fit(np.vander(xx, 11), y_smooth)
            y_fit = ridge.predict(np.vander(xx, 11))
        except LinAlgError,err:
            ridge.fit(np.vander(xx, 9), y_smooth)
            y_fit = ridge.predict(np.vander(xx, 9))
        """    
        m_data[field] = np.array(y_smooth, dtype=np.float32)

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

    
def correct_mask(data, valid_mask, fields, points = 120):
    """
    Function built to prevent high peaks when interpolating data
    It fills the first and/or the last value of the trial, setting it to 0
    if the first and/or the last value of the trial is an outlier
    """
    
    for trial in np.unique(data['Trial']):
        #Full mask of data, with True on trial
        mask_trial = data['Trial'] == trial
        
        #Masked of non outlier length of 
        valid_masked = valid_mask[mask_trial]
        
        
        #Final value of outlier vector
        if valid_masked[-1] == False:
            index = np.nonzero(mask_trial)[0][-1]
            valid_mask[index] = True
            for field in fields:
                data[field][index] = np.mean(data[field][mask_trial][valid_masked][-points:])
                mean_d = np.mean(data[field][valid_mask * mask_trial][:points])
                data[field][index] = mean_d
                
        #First value of outlier vector
        
        if valid_masked[0] == False:
            index = np.nonzero(mask_trial)[0][0]
            valid_mask[index] = True
            for field in fields:
                data[field][index] = np.mean(data[field][mask_trial][valid_masked][:points])
                
    return valid_mask