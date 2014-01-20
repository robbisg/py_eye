#######################################################
#     Copyright (c) 2013 Roberto Guidotti
#
#     See the file license.txt for copying permission.
########################################################


from py_eye.io import *
from py_eye.polish import *
from py_eye.preprocessing import *
from py_eye.analysis import *
import numpy.lib.recfunctions as nprec
import itertools
import copy

filename = 'Sub14_1.txt'

path_d = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/'
path_data = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/'
path = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/corrected/'
path_b = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data/'
path_rem_bi = '/media/DATA/eye_analysis/interp_baseline/'
path_f = '/home/robbis/eye/fitted/'
path_i = '/home/robbis/eye/interp_baseline/'
path_blink = '/home/robbis/eye/blink/'
path_bc = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural_Data_Seq/'
path_c = '/media/DATA/eye_analysis/corrected/'


conf = read_configuration(path_d, 'eye_analysis.conf')
paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
fields = conf['data_fields'].split(',')
baseline_condition = conf['baseline']

file_list = os.listdir(path)
file_list.sort()
results = []
results = dict()


max_pupil = [8.]
min_pupil = [2.]


lista = [max_pupil, min_pupil]

s = itertools.product(*lista)

for value in s:
    results = dict()
    t_count = dict()

    max = value[0]
    min = value[1]
    
    for filen in file_list:
        
        d_data = load_data_eye(path, filen)
        
        trial_info = extract_trials_info(d_data)    
        conf = read_configuration(path_d, 'eye_analysis.conf')
        paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
        
        name = filen.split('.')[0]    
        
        try:
            behavioural = open_behavioural(path_b, name+'.xlsx')
        except IOError,err:
            print str(err)
            
        trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
        
        #####################################################
          
        d_data, field = merge_fields(d_data, **conf)
        
        conf['data_fields'] = field
        conf['min_pupil_size'] = min
        conf['max_pupil_size'] = max
        
        factor = np.float(conf['pupil_factor'])
        fields = conf['data_fields'].split(',')
        
        #d_data = pixel_to_mm(d_data, fields, factor)
                
        #downsampling
        d_data['data'], d_data['SampleRate'] = split_data(d_data, fields)
        
        #Where is used??
        conf['window'] = d_data['SampleRate'] * 0.1
        conf['baseline_size'] = d_data['SampleRate'] * float(conf['baseline_size'])
        
        valid_mask = remove_outliers(d_data, **conf)
        #valid_mask = remove_outliers_gmm(d_data, **conf)
        
        
        valid_mask = correct_mask(d_data['data'], valid_mask, fields, 
                                  d_data['SampleRate'], seconds = 0.5)
        
        try:
            [i_data, definitive_mask, bad_trials] = interpolate_trial(d_data['data'], trial_info, fields, valid_mask)
        except ValueError, err:
            print str(err)
            #continue
        
        
        
        
        d_data['data'] = baseline_correction(d_data['data'], definitive_mask, trial_info, 
                                      d_data['SampleRate'], type='previous', **conf)
        
        d_data['data'] = d_data['data'][definitive_mask]
        
        #d_data['data'] = i_data[definitive_mask]
        #write_corrected(path, file, path_blink, file, d_data)
        #write_corrected(path, file, path_i, file, d_data)
    
    
        #downsampling
        
                
        trial_info = extract_trials_info(d_data)
        trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
        trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
        
        t_count[name] = count_good_trials(behavioural, trial_cond, **conf)
        
        an = analyze_timecourse(d_data['data'], trial_cond, d_data['SampleRate'], **conf)
        results[name] = an
        
        sample_rate = d_data['SampleRate']
        del d_data, i_data
        
    
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
                        time = np.arange(0, len(r_data)/sample_rate, 1/sample_rate)
                        sheet_2.write(s2, 4, float(time[r_data == np.max(r_data)][0]))
                
                    for i in range(len(r_data)):
                        if flag == 0:
                            time = np.arange(0, len(r_data)/sample_rate, 1/sample_rate)
                            sheet.write(3+i, 0, float(time[i]))
                        sheet.write(3+i, r_pos, float(r_data[i]))
                    flag = 1
    filename = 'wbook_simon_bologna_60Hz.xls'
    wbook.save('/home/robbis/eye/'+filename)
    
    filename = 'trial_count_simon_bologna_60Hz.txt'
    file_trial = open('/home/robbis/eye/'+filename, 'w')
    
    conditions = []
    try:
        conditions = np.int_(conf['conditions'].split(','))
    except ValueError, err:
        conditions = conf['conditions'].split(',')


    file_trial.write('Subj\t')
    for cond in conditions:
        cond = str(cond)
        file_trial.write('\t'+str(cond)+'_tot')
        file_trial.write('\t')
    file_trial.write('\n\r')

    for key in t_count.keys():
        file_trial.write(str(key))
        file_trial.write('\t')
        for n in t_count[key]:
            file_trial.write(str(n))
            file_trial.write('\t')
        file_trial.write('\n\r')

    file_trial.close()
    
###########################################################################
file_list = os.listdir(path_f)
file_list.sort()

results = dict()

path_b = '/home/robbis/eye/interp_baseline'
file_list = os.listdir(path_b)
file_list.sort()
file_list = file_list[1:2]

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
        behavioural = open_behavioural(path_bd, name+'.xlsx')
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
    
    an = analyze_timecourse(d_data['data'], trial_cond, d_data['SampleRate'], **conf)

    results[name] = an       
    #################################################
results = dict()
t_count = dict()   
    
for filen in filelist:
    
    d_data = load_data_eye_v2(path, filen)
    d_data = clear_trials(d_data, [0])
    trial_info = extract_trials_info(d_data)    
    conf = read_configuration(path, 'experiment.conf')
    
    name = filen.split('.')[0]    
    
    if name == 'Sub3':
        fn_para = 'LISTA_SET_s3.xlsx'
    else:
        fn_para = 'LISTA_SET_dropped_trial.xlsx'
    paradigm = read_paradigm(path, fn_para)
    
    try:
        behavioural = open_behavioural_v2(path, name+'.xls')
    except IOError,err:
        print str(err)
        
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    
    #####################################################
      
    d_data, field = merge_fields(d_data, **conf)
    
    conf['data_fields'] = field
    
    d_data['data'][field][np.isnan(d_data['data'][field])] = 0.
    '''
    conf['min_pupil_size'] = min
    conf['max_pupil_size'] = max
    
    print conf['max_pupil_size'], conf['min_pupil_size']
    '''
    factor = np.float(conf['pupil_factor'])
    fields = conf['data_fields'].split(',')
    
    #d_data = pixel_to_mm(d_data, fields, factor)
            
    #downsampling
    #d_data['data'], d_data['SampleRate'] = split_data(d_data, fields)
    
    valid_mask = remove_outliers(d_data, **conf)
    #valid_mask = remove_outliers_gmm(d_data, **conf)
    
    
    valid_mask = correct_mask(d_data['data'], valid_mask, fields, 
                              d_data['SampleRate'], seconds = 0.5)
    
    #Downsampling
    
    try:
        [i_data, definitive_mask, bad_trials] = interpolate_trial(d_data['data'], trial_info, fields, valid_mask)
    except ValueError, err:
        print str(err)
        #continue
    
    
    
    
    d_data['data'] = baseline_correction(d_data['data'], definitive_mask, trial_info, 
                                  d_data['SampleRate'], type='previous', **conf)
    
    d_data['data'] = d_data['data'][definitive_mask]
    
    #d_data['data'] = i_data[definitive_mask]
    #write_corrected(path, file, path_blink, file, d_data)
    #write_corrected(path, file, path_i, file, d_data)


    #downsampling
    
            
    trial_info = extract_trials_info(d_data)
    
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    trial_cond = trial_cond[trial_cond['Accuracy'] == 1]
    trial_cond = trial_cond[trial_cond['Length'] > 100]
    trial_cond = trial_cond[trial_cond['Trial'] > 16]
    
    t_count[name] = count_good_trials(behavioural, trial_cond, **conf)
    
    an, plot = analyze_timecourse(d_data['data'], trial_cond, d_data['SampleRate'], **conf)
    results[name] = an
    