from py_eye.io import *
from py_eye.polish import *
from py_eye.preprocessing import *
from py_eye.analysis import *
import numpy.lib.recfunctions as nprec
import itertools
import copy

root_path = '/home/robbis/Dropbox/Simon_Task_Eye_Movement/BOLOGNA/'

'''
path = os.path.join(path, 'PupillaryData','VISIVO','ALL')
path_data = os.path.join(path, 'PupillaryData')
path_behavioural = os.path.join(path, 'BehaviouralDataNew_xBin')
'''

path = os.path.join(root_path, 'ESPERIMENTO NEW WITHIN', 'VISIVO')
path_data = os.path.join(path,'PupillaryData')
path_behavioural = os.path.join(path,'BehaviouralData_xBin')

conf = read_configuration(path, 'experiment.conf')
fields = conf['data_fields'].split(',')
baseline_condition = conf['baseline']

file_list = os.listdir(path)
file_list.sort()
results = []
results = dict()
t_count = dict()
subject_list = np.loadtxt(os.path.join(path,'subject_list.txt'), 
                          dtype=np.str, 
                          delimiter=',')
    
for subj in subject_list:
    
    d_data, _ = load_data_eye_v2(path_data, subj[0])
    d_data['data'] = d_data['data'][d_data['data']['Trial']!=0]
    trial_info = extract_trials_info(d_data)    
    conf = read_configuration(path, 'experiment.conf')
    paradigm_fn = 'LISTA_TRIAL_'+subj[1]+'.xlsx'
    paradigm = read_paradigm(path, paradigm_fn, **conf)
    
    name = subj[0].split('.')[0]    
    
    dropped_trials = np.int_(conf['dropped_trials'].split(','))
    
    try:
        behavioural = open_behavioural(path_behavioural, name+'.xls', **conf)
    except IOError,err:
        print str(err)
    
    
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    
    
    #####################################################
    '''  
    d_data, field = merge_fields(d_data, **conf)
    
    conf['data_fields'] = field
    '''
    factor = np.float(conf['pupil_factor'])
    fields = conf['data_fields'].split(',')
    
    #d_data = pixel_to_mm(d_data, fields, factor)
            
    #downsampling
    #d_data['data'], d_data['SampleRate'] = split_data(d_data, fields)
    
    #Where is used??
    conf['window'] = d_data['SampleRate'] * 0.1
    #conf['baseline_size'] = d_data['SampleRate'] * float(conf['baseline_size'])
    
    valid_mask = remove_outliers(d_data, **conf)
    #valid_mask = remove_outliers_gmm(d_data, **conf)
    
    
    valid_mask = correct_mask(d_data['data'], valid_mask, fields, 
                              d_data['SampleRate'], seconds = 0.5)
    
    try:
        [i_data, definitive_mask, bad_trials] = interpolate_trial(d_data['data'], 
                                                                  trial_info, 
                                                                  fields, 
                                                                  valid_mask)
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
    ### Binseq analysis ##
    
    condition_ = np.array(behavioural[conf['behavioural_field']], dtype=np.str)
    bin_ = np.array(behavioural[conf['bin_field']], dtype=np.str)
    behavioural[conf['behavioural_field']] = np.core.defchararray.add(condition_, bin_)
    behavioural[conf['behavioural_field']] = behavioural[conf['behavioural_field']].astype(np.str)
    
    new_conditions = np.unique(behavioural[conf['behavioural_field']])
    new_conditions = np.array(new_conditions, dtype=np.str)
    new_conditions = np.array([c for c in new_conditions if str(c).find('0')==-1], np.str_)
    
    conf['conditions'] = ','.join(new_conditions)
    
    trial_info = extract_trials_info(d_data)
    trial_cond, trial_info = merge_paradigm(trial_info, paradigm, behavioural, **conf)
    
    trial_cond = trial_cond[trial_cond['correcttrial'] == 1]
    #trial_cond = trial_cond[trial_cond['imagedisplay1.acc'] == 1]
    trial_cond = trial_cond[trial_cond['Trial'] > 16]
    
    t_count[name] = count_good_trials(behavioural, trial_cond, **conf)
    
    trial_cond = trial_cond[trial_cond['Length'] > 100]

    an, _ = analyze_timecourse(d_data['data'], trial_cond, d_data['SampleRate'], **conf)
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
filename = 'wbook_simon_bologna_visivo_bin_corrtrial.xls'
wbook.save('/home/robbis/eye/'+filename)

filename = 'trial_count_simon_bologna_visivo_bin_corrtrial.txt'
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