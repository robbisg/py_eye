from py_eye.io import *
from py_eye.polish import *
from py_eye.preprocessing import *
from py_eye.analysis import *
import numpy.lib.recfunctions as nprec


filename = 'Sub14_1.txt'

path_d = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/'
path_data = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/'
path = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/corrected/'
path_b = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data/'
path_rem_bi = '/media/DATA/eye_analysis/interp_baseline/'
path_f = '/home/robbis/eye/fitted/'
#path_i = '/home/robbis/eye/interp_baseline/'
path_i = '/media/DATA/eye/interp_baseline/'
path_blink = '/home/robbis/eye/blink/'
path_bc = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data corr/'
path_c = '/media/DATA/eye_analysis/corrected/'


conf = read_configuration(path_d, 'eye_analysis.conf')
paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
fields = conf['data_fields'].split(',')
baseline_condition = conf['baseline']

file_list = os.listdir(path)
file_list.sort()
results = []
results = dict()

for file in file_list:
    
    d_data = load_data_eye(path_data, file)
    trial_info = extract_trials_info(d_data)
    errors = detect_errors(d_data, trial_info, np.array([1., 3.]))
    d_data = correct_data(d_data, trial_info, errors, np.array([1., 3.]))
    trial_info = extract_trials_info(d_data)
    file_o = file[:file.find(' ')]+'.txt'
    print file_o + ' ' + str(len(trial_info))
    
    write_corrected(path_data, file, path, file_o, d_data)
    

for file in file_list:
    
    d_data = load_data_eye(path, file)
    
    trial_info = extract_trials_info(d_data)    
    conf = read_configuration(path_d, 'eye_analysis.conf')
    paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
    
    name = file.split('.')[0]
    
    try:
        behavioural = open_behavioural(path_b, name+'.xlsx')
    except IOError,err:
        print str(err)
        
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    
    #####################################################
      
    d_data, field = merge_fields(d_data, **conf)
    
    conf['data_fields'] = field
    factor = np.float(conf['pupil_factor'])
    fields = conf['data_fields'].split(',')
    d_data = pixel_to_mm(d_data, fields, factor)
    
    valid_mask = remove_outliers(d_data, **conf)
    #valid_mask = remove_outliers_gmm(d_data, **conf)
    
    
    valid_mask = correct_mask(d_data['data'], valid_mask, fields, 120.)
    
    
    try:
        [i_data, definitive_mask] = interpolate_trial(d_data['data'], trial_info, fields, valid_mask)
    except ValueError, err:
        print str(err)
        #continue
    
    d_data['data'] = baseline_correction(d_data['data'], definitive_mask, trial_info, 
                                  type='previous', **conf)
    
    d_data['data'] = d_data['data'][definitive_mask]
    
    #d_data['data'] = i_data[definitive_mask]
    #write_corrected(path, file, path_blink, file, d_data)
    #write_corrected(path, file, path_i, file, d_data)


    #downsampling
    #an = mean_analysis(d_data['data'], trial_cond, **conf)
    trial_info = extract_trials_info(d_data)
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
    an = analyze_timecourse(d_data['data'], trial_cond, d_data['SampleRate'], **conf)
    results[name] = an



#Plot

########################################################################################
file_list = os.listdir(path_f)
file_list.sort()

results = dict()

for file in file_list:
    
    d_data = load_data_eye(path_b, file)
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
    count_file = open(path_i+'/count_trials_blink.txt', 'w')      
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
            behavioural = open_behavioural(path_b, name+'.xlsx')
        except IOError, err:
            print err
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
        count_file.write(str(np.count_nonzero(par['Condition'] == 'C')))
        count_file.write(' ')     
        count_file.write(str(np.count_nonzero(trial_cond['Condition'] == 'NC')))
        count_file.write(' ')  
        count_file.write(str(np.count_nonzero(par['Condition'] == 'NC'))) 
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
    
    d_data = load_data_eye(path_f, file)
    
    trial_info = extract_trials_info(d_data)    
    conf = read_configuration(path_d, 'eye_analysis.conf')
    paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
    
    name = file.split('.')[0]
    
    try:
        behavioural = open_behavioural(path_bc, name+'.xlsx')
    except IOError,err:
        print str(err)
        
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    
    #####################################################
    
    d_data, field = merge_fields(d_data, **conf)
    
    conf['data_fields'] = field
    
    #s_data = split_data(d_data, [field])    
    #downsampling
    #acn = analyze_timecourse(s_data, trial_cond, d_data['SampleRate'], **conf)
    trial_info = extract_trials_info(d_data)
    an = analyze_timecourse(d_data['data'], trial_cond, d_data['SampleRate'], **conf)
    del d_data#, s_data
    results[name] = an

# Write to excel
names = results.keys()
names.sort()
n_subj = len(names)
f = results[name].keys()[0]
n_fields = len(results[name].keys())
ex_cond = results[name][f].keys()[0]
n_cond = len(results[name][f].keys())
n_results = len(results[name][f][ex_cond].keys())
import xlwt

wbook = xlwt.Workbook()
wbook_2 = xlwt.Workbook()
sheet_2 = wbook_2.add_sheet('max')
sheets = dict()
for name in names:
    sheets[name] = wbook.add_sheet(name)

s2 = 0
for name, sheet in sheets.iteritems():
    sheet.write(2, 0, 'time')
    fields = results[name].keys()
    f = 0
    
    flag = 0
    for field in fields:
        f_pos = f * n_cond * n_results + 1
        sheet.write(0, f_pos, field)
        f = f + 1
        conditions = results[name][field].keys()
        c = 0
        for cond in conditions:
            c_pos = c * n_results + f_pos
            sheet.write(1, c_pos, cond)
            c = c + 1
            results_labels = results[name][field][cond].keys()
            r_col = 0
            s2 = s2 + 1
            for r in results_labels:
                r_pos = c_pos + r_col                
                sheet.write(2, r_pos, r)
                r_col = r_col + 1
                
                r_data = results[name][field][cond][r]
                
                if r == 'mean':
                    sheet_2.write(s2, 0, name)
                    sheet_2.write(s2, 1, cond)
                    sheet_2.write(s2, 2, field)
                    sheet_2.write(s2, 3, float(np.max(r_data)))
                    time = np.arange(0, len(r_data)/240., 1/240.)
                    sheet_2.write(s2, 4, float(time[r_data == np.max(r_data)][0]))
                
                for i in range(len(r_data)):
                    if flag == 0:
                        time = np.arange(0, len(r_data)/240., 1/240.)
                        sheet.write(3+i, 0, float(time[i]))
                    sheet.write(3+i, r_pos, float(r_data[i]))
                flag = 1
    
# Plot
colors = ['b','r','g','y']
    
for name in results:
    f = plt.figure()
    
    n_fields = len(results[name].keys())
    
    a = f.add_subplot(n_fields,1,n_fields)
    '''
    a.plot(results[name]['L_Dia_X'][1]['mean'])
    a.plot(results[name]['L_Dia_X'][2]['mean'])
    a.plot(results[name]['L_Dia_X'][3]['mean'])
    a.plot(results[name]['L_Dia_X'][4]['mean'])
    a.legend([1,2,3,4])
    '''
    fp = 0
    for field in results[name]:
        i = 0
        fp = fp + 1
        a = f.add_subplot(n_fields,1,fp)
        for condition in results[name][field]:
            
            mean = results[name][field][condition]['mean']
            std = results[name][field][condition]['std']
            
            x = np.arange(0, len(mean))
            
            a.plot(mean, color=colors[i])
            a.fill_between(x, mean-std, mean+std, color=colors[i], alpha=0.1)
            a.set_xlim((0, len(mean)))
            a.set_xticks(time[::69])
            i = i + 1
        a.legend(results[name][field])    
    
    f.savefig('/home/robbis/eye/'+name+'_err.png') 


################################################################
time = np.arange(0, len(r_data)/240., 1/240.)

