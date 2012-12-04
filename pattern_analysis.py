from mvpa2.suite import *  
          
def analyzeFile(datapath, attpath, subj):
    # (X, Y, trial id) for all timepoints

    data = np.loadtxt(datapath+subj, usecols = (2,9,10), skiprows = 21)

  
    newdata = [elem for elem in data if elem[0]!= 1] #Cancel first run
    newdata = np.asanyarray(newdata)
    newdata = [elem for elem in newdata if elem[1]!= 0] #Delete eyeblinks!
    newdata = np.asanyarray(newdata)
    
    
    attribs = np.loadtxt(attpath)
      
    raw_ds = Dataset(newdata[:,1:], sa = {'trial': newdata[:,0] - 1}, fa = {'fid': ['rawX', 'rawY']})
    
    print 'Dataset loaded...'
    

    #Variables to be setted
    
    npoints = 200;
    timeStimulus = 10 #sec
    nchunks =  6
    
    
    #Downsampling data to have a timeserie of 500 values.
    ds = fft_resample(raw_ds, num=npoints, window='hann', chunks_attr='trial', attr_strategy='sample')
    
    bm = BoxcarMapper(np.arange(len(ds.sa['trial'].unique)) * npoints, boxlength = npoints)
    bm.train(ds)
    ds=ds.get_mapped(bm)

    ds.sa.update({'type': attribs[:].astype(int)})
    
    fm = FlattenMapper()
    fm.train(ds)
    # want to make a copy to keep the original pristine for later plotting
    fds = ds.copy().get_mapped(fm)

    # simplify the trial attribute
    fds.sa['trial'] = [t[0] for t in ds.sa.trial]
   
    chunks = np.zeros(len(fds), dtype='int')
    for o in fds.sa['type'].unique:
        chunks[fds.sa.type == o] = np.arange(len(fds.sa.type == o)) % nchunks
        fds.sa['chunks'] = chunks
    
    
    clf = SVM(space='type')
    mclf = SplitClassifier(clf, space='type', enable_ca=['confusion'])
    
    cvte = CrossValidation(clf, NFoldPartitioner(cvtype = 1), enable_ca=['stats', 'repetition_results'])
    cv_results = cvte (fds)

    print cvte.ca.stats
    
    sensana = mclf.get_sensitivity_analyzer()
    sens = sensana(fds)
        
    
    xy_sens = fds.a.mapper[1].reverse(sens).samples

    # descriptive plots
    pl.figure(figsize = (12, 10))
    # original screen size was
    axes = ('x', 'y')
    screen_size = np.array((1280, 960))
    screen_center = screen_size / 2
    colors = ('r','b')
    fig = 1

    pl.subplot(2, 2, fig)
    pl.title('Mean Gaze Track')
    face_img = pl.imread('/home/robbis/development/eyeAnalysis/Face2F.bmp')
# determine the extend of the image in original screen coordinates
# to match with gaze position
    orig_img_extent=(screen_center[0] - face_img.shape[1]/2,
                 screen_center[0] + face_img.shape[1]/2,
                 screen_center[1] + face_img.shape[0]/2,
                 screen_center[1] - face_img.shape[0]/2)
# show face image and put it with original pixel coordinates
    pl.imshow(face_img,
          extent=orig_img_extent,
          cmap=pl.cm.gray, origin='Upper')
    pl.plot(np.mean(ds.samples[ds.sa.type == 1,:,0], axis=0),
        np.mean(ds.samples[ds.sa.type == 1,:,1], axis=0),
        colors[0], label='imagination')
    pl.plot(np.mean(ds.samples[ds.sa.type == 2,:,0], axis=0),
        np.mean(ds.samples[ds.sa.type == 2,:,1], axis=0),
        colors[1], label='perception')
    pl.axis(orig_img_extent)
    pl.legend()
    fig += 1

    pl.subplot(2, 2, fig)
    pl.title('Gaze Position X-Coordinate')
    plot_erp(ds.samples[ds.sa.type == 1,:,1], pre=0, errtype = 'std',
         color=colors[0], SR=npoints/timeStimulus)
    plot_erp(ds.samples[ds.sa.type == 2,:,1], pre=0, errtype = 'std',
         color=colors[1], SR=npoints/timeStimulus)
    pl.ylim(orig_img_extent[2:])
    pl.xlabel('Peristimulus Time')
    fig += 1

    pl.subplot(2, 2, fig)
    pl.title('Gaze Position Y-Coordinate')
    plot_erp(ds.samples[ds.sa.type == 1,:,0], pre=0, errtype = 'std',
         color=colors[0], SR=npoints/timeStimulus)
    plot_erp(ds.samples[ds.sa.type == 2,:,0], pre=0, errtype = 'std',
         color=colors[1], SR=npoints/timeStimulus)
    pl.ylim(orig_img_extent[:2])
    pl.xlabel('Peristimulus Time')
    fig += 1
    
    pl.subplot(2, 2, fig)
    pl.title('SVM-Sensitivity Profiles')
    lines = plot_err_line(xy_sens[..., 0], linestyle='-', fmt='ko', errtype='std')
    lines[0][0].set_label('X')
    lines = plot_err_line(xy_sens[..., 1], linestyle='-', fmt='go', errtype='std')
    lines[0][0].set_label('Y')
    pl.legend()
    pl.ylim((-0.01, 0.01))
    pl.xlim(0,100)
    pl.axhline(y=0, color='0.6', ls='--')
    pl.xlabel('Timepoints')

    from mvpa.base import cfg
    if cfg.getboolean('examples', 'interactive', True):
    # show all the cool figures
       #pl.show()
        figureName = '/home/robbis/development/eyeAnalysis/results/' + subj[:-4] + '.png';
        print 'Saving '+figureName
        pl.savefig(figureName)

    return figureName+'.png'