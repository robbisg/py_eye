from io import *
from polish import *
from preprocessing import *
from analysis import *
import numpy.lib.recfunctions as nprec


filename = 'Sub14_1.txt'

path_d = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/'
path_data = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/'
path = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Simon_Task.txt/corrected/'
path_b = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data/'
path_i = '/home/robbis/eye/interp/'
path_f = '/home/robbis/eye/fitted/'
path_bc = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/Behavioural Data corr/'
path_rg = '/media/DATA/eye_analysis/interp/'


file_list = os.listdir(path_rg)
file_list.sort()

conf = read_configuration(path_d, 'eye_analysis.conf')
paradigm = read_paradigm(path_d, 'LISTA_SET.xlsx')
fields = conf['data_fields'].split(',')
baseline_condition = conf['baseline']

conditions = conf['conditions'].split(',')

cond_storage = dict()

for cond in conditions:
    cond_storage[cond] = []


for file in file_list:
    
    #Se file.find(_1) fai una lista altrimenti un altra!
    
    d_data = load_data_eye(path_rg, file)
    trial_info = extract_trials_info(d_data)
    
    
    conf = read_configuration(path_d, 'eye_analysis.conf')
    fields = conf['data_fields'].split(',')
    #s_data = split_data(d_data, fields)
    
    
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
    
    trial_info = nprec.append_fields(trial_info, 
                                         'Condition', 
                                         paradigm['Condition'][mask_blink_outlier]).data
    trial_info = trial_info[trial_info['Condition'] != 'FIX']
    print len(trial_info)
    
    name = file.split('.')[0]
    
    try:
        behavioural = open_behavioural(path_b, name+'.xlsx')
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
    
    
    d_data, field = merge_fields(d_data, fields)
    
    for cond in conditions:
        
        v = gather_data(d_data, trial_cond, 'Condition', cond)
        
        cond_storage[cond].append(v)
         


for k in cond_storage.keys():
    min_c = np.min([s.shape[1] for s in cond_storage[k]])
    cond_storage[k] = [s[:,:min_c] for s in cond_storage[k]]
    cond_storage[k] = np.vstack(cond_storage[k])
    #check dimensions


results = dict()


for k in cond_storage.keys():
    results[k] = dict()
    results[k]['mean'] = np.mean(cond_storage[k], axis=0)
    results[k]['std'] = np.std(cond_storage[k], axis=0)


import xlwt

wbook = xlwt.Workbook()
sheets = dict()

sheet = wbook.add_sheet('Session_II')

k = 0
for key in results.keys():
    
    k_pos = k*2+1
    
    sheet.write(0, k_pos, key)
    
    r_pos = 0
    for r in results[key].keys():
        
        p = k_pos + r_pos
        sheet.write(1, p, r)
        row = 2
        for num in results[key][r]:
            sheet.write(row, p, float(num))
            row = row + 1
            
        r_pos = r_pos + 1
        
    k = k + 1

sheet.write(1, 0, 'time')
time = np.arange(0, len(results[key][r])/240., 1/240.)
for i in range(len(results[key][r])):
    
    sheet.write(i+2, 0, time[i])
    
    

def gather_data (d_data, trial_cond, column, cond):
    
    data = d_data['data']
    
    if cond.isdigit():
        cond = int(cond)
        
    m_cond_trial = trial_cond[column] == cond
        
    #mask_cond_data = build_mask(data, trial_cond[m_cond_trial])
    
    data_list = []
    
    for tr in trial_cond[m_cond_trial]:
        
        t_data = data['L_Dia_X'][data['Trial'] == tr['Trial']]
        data_list.append(t_data)
        
    d_min = np.min([s.size for s in data_list])
    
    data_list = np.vstack([s[:d_min] for s in data_list])
        
    return data_list
    
    
def merge_fields(d_data, fields): 
    
    data = d_data['data']
    
    s = 0
    for field in fields:
        s = s + data[field]

    data[fields[0]] = s / len(fields)
    
    d_data['data'] = data
    
    return d_data, fields[0]
        
    
    
       