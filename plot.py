import matplotlib.pylab as pl

def plot_trial(data, trial, outlier_mask=None, interp_data=None, **kwargs):
    
    for arg in kwargs:
        if arg == 'data_fields':
            fields = kwargs[arg].split(',')
            
    
    n_subplt = len(fields)
    
    f = pl.figure()
    i = 0
    for field in fields:
        
        i+=1
        a = f.add_subplot(1, n_subplt, i)
        
        mask_trial = data['Trial']==trial
        
        p_data = data[field][mask_trial]
        
        outliers = outlier_mask[mask_trial]
          
        a.plot(p_data, label='raw data')
        a.plot(outliers, label='outliers')
        
        
        id = 0
        for ip_data in interp_data:
            id+=1
            p_int_data = ip_data[field][mask_trial]
        
            a.plot(p_int_data, label='interp'+str(id))
            
        a.set_title('Trial no. '+str(trial))
        a.legend()
        
        
    
    