#############################################################################################################
#############################################################################################################
####                                Start from python console                             ####################
####          execfile('/home/robbis/development/eclipse/PyMVPA/src/stefy_script.py')                    #####
#############################################################################################################
#############################################################################################################

import os
import stefy
pathFile = '/home/robbis/development/eyeAnalysis/subj/'

listafile = os.listdir(pathFile);
listafile.sort()

listafile = [elem for elem in listafile if elem != 'Sub15.txt' and elem != 'Sub17.txt']

#datapath = '/home/robbis/development/eyeAnalysis/subj/Sub1.txt'
attpath = '/home/robbis/development/eyeAnalysis/stefy_attr.txt'

for elem in listafile:
    data = pathFile;
    print '-------------Analyzing '+elem+'  ----------------'
    result = stefy.analyzeFile(data, attpath, elem)
    print result
