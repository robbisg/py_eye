from eye_analysis.io import *
from eye_analysis.polish import *
from eye_analysis.preprocessing import *
from eye_analysis.analysis import *
import numpy.lib.recfunctions as nprec


filename = 'Sub14_1.txt'

path_d = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/'
path_data = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/'
path = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/corrected/'
path_b = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data/'
path_i = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/interp/'
path_bc = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data corr/'



conf = read_configuration(path_d, 'eye_analysis.conf')
paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
fields = conf['data_fields'].split(',')
baseline_condition = conf['baseline']

file_list = os.listdir(path)

results = []

for file in file_list:
    
    d_data = load_data_eye(path, file)
    trial_info = extract_trials_info(d_data)    
    
        
    try:
        #trial_info = np.vstack((trial_info, paradigm['Condition']))
        trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition']).data
        
    except ValueError, err:
        continue
    
    mask_task = trial_info['Condition'] != baseline_condition
        
    d_data = clear_eyeblink(d_data)
    trial_info = extract_trials_info(d_data)
    
    print 'Trials no.' + str(len(trial_info))
    
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
        
    trial_info = nprec.append_fields(trial_info, 
                                     'Condition', 
                                     paradigm['Condition'][mask_blink_outlier]).data
    mask = remove_outliers(d_data, **conf)

    data_bl = baseline_correction(d_data['data'], mask, trial_info, fields,
                                  baseline_condition, type='previous')
    
    #print data_bl.shape
    try:
        [i_data, valid_mask] = interpolate_trial(data_bl, trial_info, fields, mask)
    except ValueError, err:
        continue
    #print np.unique(i_data['Trial'])
    
    d_data['data'] = i_data[valid_mask]
    write_corrected(path, file, path_i, file, d_data)
    

    name = file.split('.')[0]
    behavioural = open_behavioural(path_bc, name+'.xlsx')
    
    m = mask_task * mask_blink_outlier
    m = m[1::2]
    
    trial_task_info = trial_info[trial_info['Condition'] != baseline_condition]
    
    trial_cond = nprec.append_fields(trial_task_info,
                                     ['Accuracy', 'Combination'], 
                                     [behavioural['Accuracy'][m], 
                                     behavioural['Combination'][m]]).data
                            
                            
                            
    trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
    
    
    #downsampling
    an = mean_analysis(i_data, trial_cond, **conf)
    
    results[name] = an
    
    #vec = np.hstack((np.array([name]), an.flatten()))
    #results.append(vec)


results_arr = np.array(results)

np.savetxt(path_data+'results_2.txt', results_arr, 
           fmt='%s %s %s %s %s %s %s %s %s\r\n')

#Plot








file_list = os.listdir(path_i)
file_list.sort()

results = dict()

for file in file_list:
    d_data = load_data_eye(path_i, file)
    trial_info = extract_trials_info(d_data)
    
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
    
    trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition'][mask_blink_outlier]).data
    
    
    name = file.split('.')[0]
    behavioural = open_behavioural(path_bc, name+'.xlsx')
    
    paradigm = paradigm[paradigm['Condition']!='FIX']
    
    m = mask_blink_outlier[1::2]
    task_trial = trial_info[trial_info['Condition'] != baseline_condition]
    
    trial_cond = nprec.append_fields(task_trial,
                                     ['Accuracy', 'Combination'], 
                                     [behavioural['Accuracy'][m], 
                                     behavioural['Combination'][m]]).data
                            
                            
                            
    trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
    
    an = mean_analysis(d_data['data'], trial_cond, **conf)

    results[name] = an


names = results.keys()
names.sort()

import xlwt

wbook = xlwt.Workbook()

sheet = results[name].keys()
sheets = dict()

for sh in sheet:
    sheets[sh] = wbook.add_sheet(sh)
    
for key,sheet in sheets.iteritems():
    for c in range(len(names)):
        sheet.write(c, 0, names[c])
        subj = names[c]
        
        conditions = results[subj][key]
        for cond, r in results[subj][key].iteritems():
            sheets[key].write(c, cond*2 - 1 , r['mean'])
            sheets[key].write(c, cond*2     , r['std'])
            

 

file_list = os.listdir(path_i)
file_list.sort()

results = []

for file in file_list:
    d_data = load_data_eye(path_i, file)
    trial_info = extract_trials_info(d_data)
    
    mask_blink = np.in1d(paradigm['Trial'], trial_info.T[0])
    
    behavioural = open_behavioural(path_b, name+'.xlsx')
    
    paradigm = paradigm[paradigm['Condition']!='FIX']
    
    m = mask_blink[1::2]
    
    
    



'''
#Plot to do
- Raw data
- outlier
- baseline correction
- tracking perido
- interpolation
- smoothed data

Look at average
'''