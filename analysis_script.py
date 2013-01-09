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
path_i = '/home/robbis/eye/interp/'
path_f = '/home/robbis/eye/fitted/'
path_bc = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data corr/'
path_c = '/media/DATA/eye_analysis/corrected/'


conf = read_configuration(path_d, 'eye_analysis.conf')
paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
fields = conf['data_fields'].split(',')
baseline_condition = conf['baseline']

file_list = os.listdir(path)

results = []
results = dict()

#file_list = ['Sub10_1.txt', 'Sub19_1.txt']

for file in file_list:
    
    d_data = load_data_eye(path, file)
    
    trial_info = extract_trials_info(d_data)    
    
        
    try:
        #trial_info = np.vstack((trial_info, paradigm['Condition']))
        trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition']).data
        
    except ValueError, err:
        #continue
        print err
    
    mask_task = trial_info['Condition'] != baseline_condition
        
    #d_data = clear_eyeblink(d_data)
    trial_info = extract_trials_info(d_data)
    
    print 'Trials no.' + str(len(trial_info))
    
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
        
    trial_info = nprec.append_fields(trial_info, 
                                     'Condition', 
                                     paradigm['Condition'][mask_blink_outlier]).data
    valid_mask = remove_outliers(d_data, **conf)

    data_bl = baseline_correction(d_data['data'], valid_mask, trial_info, fields,
                                  baseline_condition, type='previous')
    
    
    valid_mask = correct_mask(d_data['data'], valid_mask, fields)
    #print data_bl.shape
    try:
        [i_data, definitive_mask] = interpolate_trial(data_bl, trial_info, fields, valid_mask)
    except ValueError, err:
        continue
    #print np.unique(i_data['Trial'])
    
    d_data['data'] = i_data[definitive_mask]
    write_corrected(path, file, path_i, file, d_data)
    

    name = file.split('.')[0]
    
    try:
        behavioural = open_behavioural(path_bc, name+'.xlsx')
    except IOError,err:
        print str(err)
        #continue
    
    m = mask_task * mask_blink_outlier
    m = m[1::2]
    
    trial_task_info = trial_info[trial_info['Condition'] != baseline_condition]
    
    trial_cond = nprec.append_fields(trial_task_info,
                                     ['Accuracy', 'Combination'], 
                                     [behavioural['Accuracy'][m], 
                                     behavioural['Combination'][m]]).data
                            
                            
                            
    trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
    
    
    #downsampling
    an = mean_analysis(d_data['data'], trial_cond, **conf)
    
    results[name] = an
    
    #vec = np.hstack((np.array([name]), an.flatten()))
    #results.append(vec)


results_arr = np.array(results)

np.savetxt(path_data+'results_2.txt', results_arr, 
           fmt='%s %s %s %s %s %s %s %s %s\r\n')

#Plot

########################################################################################
file_list = os.listdir(path_f)
file_list.sort()

results = dict()

for file in file_list:
    
    d_data = load_data_eye(path_f, file)
    trial_info = extract_trials_info(d_data)
    #print np.max(trial_info['Trial'])
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
    
    trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition'][mask_blink_outlier]).data
    trial_info = trial_info[trial_info['Condition'] != 'FIX']
    print len(trial_info)
    
    name = file.split('.')[0]
    
    try:
        behavioural = open_behavioural(path_bc, name+'.xlsx')
    except IOError, err:
        print str(err)
        #continue
    
    #paradigm = paradigm[paradigm['Condition']!='FIX']
    
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
        i = 0
        for cond, r in results[subj][key].iteritems():
            i = i + 1
            sheets[key].write(c, i*2 - 1 , r['mean'])
            sheets[key].write(c, i*2     , r['std'])
            
##############################################################################à
def write_deleted_trials(path_i):   
    del_file = open(path_d+'/del_trials_number.txt', 'w')      
 
    for file in file_list:
        d_data = load_data_eye(path_i, file)
        trial_info = extract_trials_info(d_data)
        d = []
        mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
        d = np.nonzero(~mask_blink_outlier)[0] + 2
        
        del_file.write(file+' '+str(len(d))+' \r\n')       

    del_file.close()
    
    
def count_good_trials():
    count_file = open(path_i+'count_trials.txt', 'w')      
    count_file.write('Subj C_inc C_tot  NC_inc NC_tot 1_inc 1_tot 2_inc 2_tot 3_inc 3_tot 4_inc 4_tot\r\n')
    for file in file_list:
        d_data = load_data_eye(path_i, file)
        trial_info = extract_trials_info(d_data)
        mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
        trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition'][mask_blink_outlier]).data
        task_trial = trial_info[trial_info['Condition'] != 'FIX']
        name = file.split('.')[0]
        try:
            behavioural = open_behavioural(path_bc, name+'.xlsx')
        except IOError, err:
            print str(err)
            continue
        m = mask_blink_outlier[1::2]
        trial_cond = nprec.append_fields(task_trial,
                                     ['Accuracy', 'Combination'], 
                                     [behavioural['Accuracy'][m], 
                                     behavioural['Combination'][m]]).data
        par = nprec.append_fields(paradigm[1::2],
                                     'Accuracy', 
                                     behavioural['Accuracy']).data
        trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
        par = par[par['Accuracy'] == 1]
        behavioural = behavioural[behavioural['Accuracy'] == 1]
        count_file.write(file+' ')
        count_file.write(str(np.count_nonzero(trial_cond['Condition'] == 'C')))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(paradigm['Condition'] == 'C')))
        count_file.write(' ')     
        count_file.write(str(np.count_nonzero(trial_cond['Condition'] == 'NC')))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(paradigm['Condition'] == 'NC'))) 
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(trial_cond['Combination'] == 1)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(behavioural['Combination'] == 1)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(trial_cond['Combination'] == 2)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(behavioural['Combination'] == 2)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(trial_cond['Combination'] == 3)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(behavioural['Combination'] == 3)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(trial_cond['Combination'] == 4)))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(behavioural['Combination'] == 4)))
        count_file.write(' \r\n')
          
    count_file.close()        
        
        
        
###############################################################################

file_list = os.listdir(path_i)
file_list.sort()
trial_expected = np.array([1. , 3.])

path_c = '/media/DATA/eye_analysis/corrected'
path_i = '/media/DATA/eye_analysis/interp/'

for file in file_list:
    d_data = load_data_eye(path_data, file)
    trial_info = extract_trials_info(d_data)
    errors = detect_errors(d_data, trial_info, trial_expected)
    d_data = correct_data(d_data, trial_info, errors, trial_expected)

    file_o = file.split(' ')[0]+'.txt'
    write_corrected(path_data, file, path_c, file_o, d_data)

######################################################################
file_list = os.listdir(path_i)
file_list.sort()

results = dict()

for file in file_list:
    
    d_data = load_data_eye(path_i, file)
    trial_info = extract_trials_info(d_data)
    
    
    conf = read_configuration(path_d, 'eye_analysis.conf')
    fields = conf['data_fields'].split(',')
    s_data = split_data(d_data, fields)
    
    
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
    
    trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition'][mask_blink_outlier]).data
    trial_info = trial_info[trial_info['Condition'] != 'FIX']
    print len(trial_info)
    
    name = file.split('.')[0]
    
    try:
        behavioural = open_behavioural(path_bc, name+'.xlsx')
    except IOError, err:
        print str(err)
        continue
    
    #paradigm = paradigm[paradigm['Condition']!='FIX']
    
    m = mask_blink_outlier[1::2]
    task_trial = trial_info[trial_info['Condition'] != baseline_condition]
    
    trial_cond = nprec.append_fields(task_trial,
                                     ['Accuracy', 'Combination'], 
                                     [behavioural['Accuracy'][m], 
                                     behavioural['Combination'][m]]).data
                            
                            
                            
    trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
    
    an = mean_analysis(d_data['data'], trial_cond, **conf)

    results[name] = an   

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