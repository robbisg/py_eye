import os
import numpy as np
import ConfigParser
from polish import clear_first_trials

def load_data_eye(path, filename):
    
    name = os.path.join(path, filename)
    
    hdr = open(name, 'r')
    
    i = 0
    line = 'on'
    print 'Loading '+ filename
    while i <= 19 :
        line = hdr.readline().split()
        i += 1
        if (line.count('Rate:') != 0):
            fs = float(line[len(line)-1])
        if (line.count('Area:')!= 0):
            dim_x = int(line[len(line)-2])
            dim_y = int(line[len(line)-1])
        if (line.count('Stimulus')!= 0):
            img_x = int(line[len(line)-2])
            img_y = int(line[len(line)-1])
    
    
    data = np.genfromtxt(name,
                        dtype={'names': ['Time','Type','Trial',
                                        'L Raw X',' L Raw Y','L Dia X',
                                        'L Dia Y','L CR1 X','L CR1 Y',
                                        'L POR X','L POR Y','Timing','Latency'],                      
                               'formats': ['f4', 'S4',  'f4', 'f4', 'f4',
                                         'f4', 'f4', 'f4', 'f4',
                                         'f4', 'f4', 'f4', 'f4']    
                               },
                #         filling_values = 'SMP',
                        #converters = {1: lambda s: float(s.count('') or 0)},
              #          usecols = (0,1,2,3,4,5,6,7,8,9,10,11,12),
                        skip_header = 21)
    
    
    #data_col = ['Time','Type','Trial',
    #            'L Raw X',' L Raw Y','L Dia X',
    #            'L Dia Y','L CR1 X','L CR1 Y',
    #            'L POR X','L POR Y','Timing','Latency']
    
    value = [fs, dim_x, dim_y, img_x, img_y, data]
    keys  = ['SampleRate', 'ScreenX', 'ScreenY', 'StimX', 'StimY', 'data']
    
    d_data = dict(zip(keys, value))
    
    d_data = clear_first_trials(d_data)

    return d_data


def write_corrected(pathI, filenameI, pathO, filenameO, d_data):
    
    i_name = os.path.join(pathI,filenameI)
    o_name = os.path.join(pathO, filenameO)
    
    
    inputfile = open(i_name, 'r')
    outputfile = open(o_name, 'w')
    
    i = 0
    
    for i in range(20):
        l_in = inputfile.readline()
        outputfile.write(l_in)
    
    fmt = ['%d','%s', '%d', '%4.4f', '%4.4f','%2.6f',
          '%2.6f', '%4.4f', '%4.4f', '%4.4f','%4.4f', '%d', '%d']
      
    np.savetxt(outputfile, d_data['data'], fmt=fmt, delimiter='\t', newline='\r\n')
    
    inputfile.close()
    outputfile.close()

def read_configuration(path, conf_path):
    
    config = ConfigParser.ConfigParser()
    
    config.read(os.path.join(path, conf_path))
    
    
    print 'Reading config file '+os.path.join(path, conf_path)
    
    #types = config.get('path', 'types').split(',')
    
    #if types.count(type) > 0:
    #    types.remove(type)
    
    #for typ in types:
    #    config.remove_section(typ)
    
    configuration = []
    
    for sec in config.sections():
        
        for item in config.items(sec):
            configuration.append(item)
            print item
    
    return dict(configuration)   


def read_paradigm(path, filename):
    
    ext = filename.split('.')[1]

    if ext.find('xlsx') != -1:
        paradigm = read_xls_paradigm(path, filename)
    elif ext.find('txt') != -1:
        paradigm = read_txt(path, filename)
    else:
        print ('Can\'t read '+ext+' files!')
        return 1    
    
    return paradigm 

def read_xls_paradigm (path, filename):
    
    import xlrd
    
    fn = os.path.join(path, filename)
    
    book = xlrd.open_workbook(fn)
    sh = book.sheet_by_index(0)
    
    paradigm = np.array(zip(sh.col_values(0)[1:], 
                            np.int_(sh.col_values(1)[1:])), 
                        dtype=[('Condition', np.str_,4),
                                ('Trial', np.int_, 1)])
    
    return paradigm

    
def extract_trials_info(d_data):
    
    data = d_data['data']
    
    trial_info = []
    
    for trial in np.unique(data['Trial']):
        mask_trial = data['Trial'] == trial
        trial_d = data[mask_trial]
      #  print str(trial)+' '+str(len(trial_d.T[0]))+ \
      #  ' '+str(trial_d.T[0][len(trial_d.T[0])-1] - trial_d.T[0][0] )
        
        trial_info.append((trial, len(trial_d['Time']), \
                           trial_d['Time'][len(trial_d['Time'])-1] - trial_d['Time'][0]))
    
    dt = np.dtype([('Trial', np.int16) , ('Length', np.int32), ('Index', np.int32)])
       
    return np.array(trial_info, dtype=dt)    
    
    
def check_deleted_trials(d_data, trial_info, paradigm):
    
    mask_blink_outlier = np.in1d(paradigm['Trial'], trial_info['Trial'])
    
    return paradigm['Trial'][True - mask_blink_outlier]
    
    
    
def write_to_excel(result, filename):
    '''
    To be written
    '''
    
    
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
    
    
    
    
    