Search.setIndex({docnames:["Ingest_redmagic","auxiliary_maps","base_stage","blinding","convergence","covariance","diagnostics","exposure_info","index","input_cats","installation","lens_selector","map_correlations","map_plots","maps","masks","metacal_gcr_input","metadata","noise_maps","photoz","photoz_mlz","photoz_stack","psf_diagnostics","random_cats","source_selector","stages","twopoint","twopoint_fourier","utils"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["Ingest_redmagic.rst","auxiliary_maps.rst","base_stage.rst","blinding.rst","convergence.rst","covariance.rst","diagnostics.rst","exposure_info.rst","index.rst","input_cats.rst","installation.rst","lens_selector.rst","map_correlations.rst","map_plots.rst","maps.rst","masks.rst","metacal_gcr_input.rst","metadata.rst","noise_maps.rst","photoz.rst","photoz_mlz.rst","photoz_stack.rst","psf_diagnostics.rst","random_cats.rst","source_selector.rst","stages.rst","twopoint.rst","twopoint_fourier.rst","utils.rst"],objects:{"txpipe.TXAuxiliaryMaps":{accumulate_maps:[1,1,1,""],choose_pixel_scheme:[1,1,1,""],data_iterator:[1,1,1,""],finalize_mappers:[1,1,1,""],name:[1,2,1,""],prepare_mappers:[1,1,1,""]},"txpipe.base_stage":{PipelineStage:[2,0,1,""]},"txpipe.base_stage.PipelineStage":{open_output:[2,1,1,""]},"txpipe.blinding":{TXBlinding:[3,0,1,""],TXNullBlinding:[3,0,1,""]},"txpipe.blinding.TXBlinding":{run:[3,1,1,""]},"txpipe.blinding.TXNullBlinding":{run:[3,1,1,""]},"txpipe.convergence":{TXConvergenceMapPlots:[4,0,1,""],TXConvergenceMaps:[4,0,1,""]},"txpipe.covariance":{TXFourierGaussianCovariance:[5,0,1,""],TXRealGaussianCovariance:[5,0,1,""]},"txpipe.diagnostics":{TXDiagnosticPlots:[6,0,1,""]},"txpipe.exposure_info":{TXExposureInfo:[7,0,1,""]},"txpipe.ingest_redmagic":{TXIngestRedmagic:[0,0,1,""]},"txpipe.input_cats":{TXBuzzardMock:[9,0,1,""],TXCosmoDC2Mock:[9,0,1,""],make_mock_photometry:[9,4,1,""]},"txpipe.input_cats.TXCosmoDC2Mock":{load_metacal_response_model:[9,1,1,""],make_mock_metacal:[9,1,1,""],remove_undetected:[9,1,1,""],write_output:[9,1,1,""]},"txpipe.lens_selector":{TXBaseLensSelector:[11,0,1,""],TXMeanLensSelector:[11,0,1,""],TXTruthLensSelector:[11,0,1,""]},"txpipe.lens_selector.TXBaseLensSelector":{run:[11,1,1,""],select_lens:[11,1,1,""],setup_output:[11,1,1,""],write_global_values:[11,1,1,""],write_tomography:[11,1,1,""]},"txpipe.map_correlations":{TXMapCorrelations:[12,0,1,""]},"txpipe.map_plots":{TXMapPlots:[13,0,1,""]},"txpipe.maps":{TXBaseMaps:[14,0,1,""],TXDensityMaps:[14,0,1,""],TXExternalLensMaps:[14,0,1,""],TXLensMaps:[14,0,1,""],TXMainMaps:[14,0,1,""],TXSourceMaps:[14,0,1,""]},"txpipe.maps.TXBaseMaps":{accumulate_maps:[14,1,1,""],choose_pixel_scheme:[14,1,1,""],data_iterator:[14,1,1,""],finalize_mappers:[14,1,1,""],prepare_mappers:[14,1,1,""],save_maps:[14,1,1,""]},"txpipe.maps.TXExternalLensMaps":{data_iterator:[14,1,1,""]},"txpipe.maps.TXLensMaps":{accumulate_maps:[14,1,1,""],data_iterator:[14,1,1,""],finalize_mappers:[14,1,1,""],prepare_mappers:[14,1,1,""]},"txpipe.maps.TXMainMaps":{data_iterator:[14,1,1,""],finalize_mappers:[14,1,1,""],prepare_mappers:[14,1,1,""]},"txpipe.maps.TXSourceMaps":{accumulate_maps:[14,1,1,""],data_iterator:[14,1,1,""],finalize_mappers:[14,1,1,""],prepare_mappers:[14,1,1,""]},"txpipe.masks":{TXSimpleMask:[15,0,1,""]},"txpipe.metacal_gcr_input":{TXIngestStars:[16,0,1,""],TXMetacalGCRInput:[16,0,1,""]},"txpipe.metadata":{TXTracerMetadata:[17,0,1,""]},"txpipe.noise_maps":{TXExternalLensNoiseMaps:[18,0,1,""],TXNoiseMaps:[18,0,1,""],TXSourceNoiseMaps:[18,0,1,""]},"txpipe.noise_maps.TXExternalLensNoiseMaps":{accumulate_maps:[18,1,1,""],choose_pixel_scheme:[18,1,1,""],data_iterator:[18,1,1,""],finalize_mappers:[18,1,1,""],prepare_mappers:[18,1,1,""]},"txpipe.noise_maps.TXSourceNoiseMaps":{accumulate_maps:[18,1,1,""],choose_pixel_scheme:[18,1,1,""],data_iterator:[18,1,1,""],finalize_mappers:[18,1,1,""],prepare_mappers:[18,1,1,""]},"txpipe.photoz":{TXRandomPhotozPDF:[19,0,1,""]},"txpipe.photoz.TXRandomPhotozPDF":{calculate_photozs:[19,1,1,""],prepare_output:[19,1,1,""],run:[19,1,1,""],write_output:[19,1,1,""]},"txpipe.photoz_mlz":{PZPDFMLZ:[20,0,1,""]},"txpipe.photoz_mlz.PZPDFMLZ":{calculate_photozs:[20,1,1,""],prepare_output:[20,1,1,""],run:[20,1,1,""],write_output:[20,1,1,""]},"txpipe.photoz_stack":{TXPhotozPlots:[21,0,1,""],TXPhotozSourceStack:[21,0,1,""],TXPhotozStack:[21,0,1,""],TXSourceTrueNumberDensity:[21,0,1,""],TXTrueNumberDensity:[21,0,1,""]},"txpipe.photoz_stack.TXPhotozSourceStack":{get_metadata:[21,1,1,""],run:[21,1,1,""]},"txpipe.photoz_stack.TXPhotozStack":{get_metadata:[21,1,1,""]},"txpipe.photoz_stack.TXSourceTrueNumberDensity":{get_metadata:[21,1,1,""]},"txpipe.photoz_stack.TXTrueNumberDensity":{get_metadata:[21,1,1,""]},"txpipe.psf_diagnostics":{TXBrighterFatterPlot:[22,0,1,""],TXPSFDiagnostics:[22,0,1,""],TXRoweStatistics:[22,0,1,""],TXStarDensityTests:[22,0,1,""],TXStarShearTests:[22,0,1,""]},"txpipe.random_cats":{TXRandomCat:[23,0,1,""]},"txpipe.source_selector":{TXSourceSelector:[24,0,1,""]},"txpipe.source_selector.TXSourceSelector":{apply_classifier:[24,1,1,""],calculate_tomography:[24,1,1,""],run:[24,1,1,""],setup_output:[24,1,1,""],write_global_values:[24,1,1,""],write_tomography:[24,1,1,""]},"txpipe.twopoint":{Measurement:[26,0,1,""],TXGammaTBrightStars:[26,0,1,""],TXGammaTDimStars:[26,0,1,""],TXGammaTFieldCenters:[26,0,1,""],TXGammaTRandoms:[26,0,1,""],TXJackknifeCenters:[26,0,1,""],TXTwoPoint:[26,0,1,""],TXTwoPointLensCat:[26,0,1,""],TXTwoPointPlots:[26,0,1,""]},"txpipe.twopoint.Measurement":{__getnewargs__:[26,1,1,""],__new__:[26,1,1,""],__repr__:[26,1,1,""],corr_type:[26,2,1,""],i:[26,2,1,""],j:[26,2,1,""],object:[26,2,1,""]},"txpipe.twopoint.TXGammaTBrightStars":{read_nbin:[26,1,1,""],run:[26,1,1,""]},"txpipe.twopoint.TXGammaTDimStars":{read_nbin:[26,1,1,""],run:[26,1,1,""]},"txpipe.twopoint.TXGammaTFieldCenters":{read_nbin:[26,1,1,""],run:[26,1,1,""]},"txpipe.twopoint.TXGammaTRandoms":{read_nbin:[26,1,1,""],run:[26,1,1,""]},"txpipe.twopoint.TXJackknifeCenters":{plot:[26,1,1,""]},"txpipe.twopoint.TXTwoPoint":{call_treecorr:[26,1,1,""],get_calibrated_catalog_bin:[26,1,1,""],read_nbin:[26,1,1,""],run:[26,1,1,""]},"txpipe.twopoint.TXTwoPointPlots":{get_theta_xi_err:[26,1,1,""],get_theta_xi_err_jk:[26,1,1,""]},"txpipe.twopoint_fourier":{Measurement:[27,0,1,""],TXTwoPointFourier:[27,0,1,""],TXTwoPointPlotsFourier:[27,0,1,""]},"txpipe.twopoint_fourier.Measurement":{__getnewargs__:[27,1,1,""],__new__:[27,1,1,""],__repr__:[27,1,1,""],corr_type:[27,2,1,""],i:[27,2,1,""],j:[27,2,1,""],l:[27,2,1,""],value:[27,2,1,""],win:[27,2,1,""]},"txpipe.utils":{pixel_schemes:[28,3,0,"-"]},"txpipe.utils.pixel_schemes":{choose_pixelization:[28,4,1,""],round_approx:[28,4,1,""]},txpipe:{TXAuxiliaryMaps:[1,0,1,""],base_stage:[2,3,0,"-"],blinding:[3,3,0,"-"],convergence:[4,3,0,"-"],covariance:[5,3,0,"-"],diagnostics:[6,3,0,"-"],exposure_info:[7,3,0,"-"],ingest_redmagic:[0,3,0,"-"],input_cats:[9,3,0,"-"],lens_selector:[11,3,0,"-"],map_correlations:[12,3,0,"-"],map_plots:[13,3,0,"-"],maps:[14,3,0,"-"],masks:[15,3,0,"-"],metacal_gcr_input:[16,3,0,"-"],metadata:[17,3,0,"-"],noise_maps:[18,3,0,"-"],photoz:[19,3,0,"-"],photoz_mlz:[20,3,0,"-"],photoz_stack:[21,3,0,"-"],psf_diagnostics:[22,3,0,"-"],random_cats:[23,3,0,"-"],source_selector:[24,3,0,"-"],twopoint:[26,3,0,"-"],twopoint_fourier:[27,3,0,"-"]}},objnames:{"0":["py","class","Python class"],"1":["py","method","Python method"],"2":["py","attribute","Python attribute"],"3":["py","module","Python module"],"4":["py","function","Python function"]},objtypes:{"0":"py:class","1":"py:method","2":"py:attribute","3":"py:module","4":"py:function"},terms:{"abstract":14,"byte":[],"case":[26,28],"class":[0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],"default":28,"final":[1,14,18],"float":28,"function":[9,19,21],"int":[11,19,20,24,28],"long":9,"new":[26,27],"null":3,"return":[1,2,14,18,19,20,21,26,27,28],"static":[26,27],"true":21,"var":14,For:[2,26],Not:22,The:[10,11,19,20,24,27],These:25,Use:9,Used:[26,27],Uses:21,__delattr__:[],__dir__:[],__eq__:[],__format__:[],__ge__:[],__getattribute__:[],__getnewargs__:[26,27],__gt__:[],__hash__:[],__le__:[],__lt__:[],__ne__:[],__new__:[26,27],__reduce__:[],__reduce_ex__:[],__repr__:[26,27],__setattr__:[],__sizeof__:[],__str__:[],__subclasscheck__:[],__subclasshook__:[],__weakref__:[],_cl:[26,27],abc:[],abcmeta:[],about:17,abov:[10,14],access:10,accord:24,accumul:[11,21,24],accumulate_map:[1,14,18],accur:[],actual:[17,19,20],add:24,added:9,advanc:[9,16],after:[22,26,27],aggreg:[],algorithm:11,alia:[26,27],all:[4,10,13,19,20,27],alloc:21,also:[2,9,10,16,24],analysi:[3,11,19,20,21,24,26],ani:[1,9,14,18,19,20],anyth:17,appli:[19,24],apply_classifi:24,arg:[0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],around:17,arrai:[9,11,19,20,21,24,28],assum:[9,19,20],astr511:9,astropi:10,auto:27,autocorrel:22,auxilliari:25,avail:[9,13,16],averag:[11,24],axi:[19,20],band:9,barnei:22,base:[0,1,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28],base_stag:[0,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],basic:14,becaus:[14,22],being:[9,16],between:[],bia:[11,24],bias:[11,24],bin:[9,11,19,20,21,24,26,27],blind:25,block:26,bool:28,boss:11,boss_galaxy_t:11,both:[21,27],bright:1,cach:[],calcul:[17,19,26],calculate_photoz:[19,20],calculate_tomographi:24,calibr:[11,24],call:22,call_treecorr:26,can:[1,10,14,18,19],catalog:[14,16,17,19,20,24,25],ceci:[2,10],center:26,check:28,choic:[24,28],choose_pixel:28,choose_pixel_schem:[1,14,18],chunk:[1,11,14,18,19,20,24],classifi:24,close:[11,19,24,28],cluster:11,code:10,collat:17,collect:[11,24],column:[11,14,21,24],combin:[14,27],come:27,comm:[0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],command:[],commun:[],complic:9,comput:[19,20,27],config:28,configur:[2,9,24,26,28],connect:[],consist:24,construct:[24,28],contain:[9,19,20,24],content:8,converg:4,convergencemap:25,convert:14,copi:[3,26,27],cori:10,corr_typ:[26,27],correct:26,correl:[25,27],cosmo:9,cosmolog:[9,16],count:[14,21],covari:[25,26],creat:[1,11,14,18,24,26,27],cross:[22,27],current:[27,28],custom:[],cut:[11,24],data:[1,3,9,10,11,14,16,18,19,20,21,24,26],data_iter:[1,14,18],datapoint:26,datayp:26,dc2:[9,16],dec:26,decid:9,defin:28,delattr:[],deletet:3,delta:[14,21],densiti:[14,22],depend:10,deproject:27,depth:1,describ:14,design:[19,20],detail:[],detect:9,determin:26,deviat:9,diagnost:25,dict:[9,19,20,24],dictionari:[1,9,14,18],differ:[19,21],dir:[],directli:14,disc:9,divid:21,document:[],doe:[19,20,21],doesn:17,domin:27,don:17,done:[19,20],down:28,dr9:11,each:[11,14,19,20,21,24],earli:[],edu:9,either:26,ell:27,elsewher:14,end:[11,19,20,24],environ:10,err:26,errorbar:26,estim:[19,20],evalu:[19,20],exampl:10,except:[14,28],execut:[],exist:[1,9,14,18],exposur:26,exposure_info:25,extend:2,extern:[14,26],factor:[11,24,26],faculti:9,fake:21,fals:[2,28],far:[9,25],faster:14,featur:[20,24],field:[26,27],file:[2,3,9,10,11,19,20,21,24,26],finalize_mapp:[1,14,18],financ:22,find:2,first:22,fit:2,fitsio:10,five:[19,20],flag:[1,24,27],floor:28,follow:27,format:[19,20,26,27],formatt:[],former:10,from:[1,9,10,14,16,18,19,20,24,26,27],full:[19,20],futur:27,galaxi:[11,14,18,22,27],gener:[1,2,9,14,18,19,20,24,26,27],get:[21,26],get_calibrated_catalog_bin:26,get_metadata:21,get_theta_xi_err:26,get_theta_xi_err_jk:26,getattr:[],give:24,given:[2,26],global:10,gov:10,group:[10,11,24],h5py:[11,19,20,24],handi:[9,16],handl:[],has:14,hash:[],have:[9,10,19],hdf5:[2,19,20],hdf:9,healpi:10,healpix:28,help:[],helper:[],here:[2,9,19],high:22,histogram:21,howev:[19,20],http:[9,11],imag:[9,16],implement:[],includ:27,incorrig:22,index:[8,11,19,20,24],individu:18,info:2,inform:[2,24],infrastructur:[9,16],ingest_redmag:0,inherit:14,init:[1,14,18],input:[1,10,14,16,18,19,20,21,24,25,26,27],input_cat:9,instal:8,instanc:[26,27,28],instead:[1,14,18,19,20,26],integ:28,interact:26,invok:[],issubclass:[],iter:[1,11,14,18,21,24],its:[9,21,26],ivez:9,jackknif:26,job:10,jone:9,just:[9,17,19,20],kappa_:4,kappa_b:4,kind:[9,16],kwarg:[2,28],later:[14,19,20],latter:10,len:[14,21,25,26,27],lens:[14,26,27],lens_bin:11,lens_selector:11,lensfit:24,letter:22,like:2,limit:9,line:[],list:[11,24,27],load:[1,3,9,14,18,19,21,26],load_metacal_response_model:9,log10:9,log:19,login:10,loop:[1,11,14,18,19,21,24],lost:22,lse:9,lsst:10,lsst_snrdoc:9,lupton:9,lynn:9,machin:10,made:[14,24],magnitud:[19,20,24],mainli:[9,16],make:[4,13,14,21,24,26],make_mock_metac:9,make_mock_photometri:9,map:[4,13,18,25,27],map_correl:25,map_nam:[1,14,18],map_plot:25,mapper:[1,14,18],mask:[14,25,26],matric:24,mean:[9,19,20,24],measur:[9,16,18,24,26,27],memori:[],meta:26,metac:[9,16,19,24,26],metacal_col:9,metacal_data:9,metacal_fil:9,metacal_gcr_input:25,metacalibr:[9,16,19,20,24],metadata:[21,25],method:[2,19,20,21,26],might:[9,16,19],mnra:22,mock:[9,19,20],mode:[2,27],model:9,modul:[8,11,24],moment:19,more:2,mostli:9,move:19,mpi:[],much:[17,24],multipl:[11,24],must:[1,14,18,19,28],n_chunk:[19,20],n_gal:14,n_visit:9,n_z:[19,20],name:[1,22],nan:26,nbin:[21,24],need:[10,17,19,24],nersc:10,nest:28,next:[1,14,18],ngal:14,nice:[26,27],nobj:[19,20],node:10,nois:[9,18],noise_map:25,nompi:10,none:[0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],normal:19,note:[19,20],notimpl:[],now:[2,14,22],nrow:[11,24],nside:28,number:[14,19,20,21,26,27],number_density_stat:[11,24],numpi:10,object:[1,2,9,11,14,17,18,19,20,21,24,26,27],obs:[9,16],obtain:10,onc:[14,19,20,24],one:[9,19],onli:[9,14,28],open:[2,19,20],open_output:2,option:[3,24,28],order:28,org:11,other:14,otherwis:[],our:17,out:[9,11,19,20,24,28],outcom:[],outfil:[11,24],output:[2,3,11,19,20,21,24],output_fil:[19,20],output_tag:[1,14,18],over:[1,14,18],overal:[11,24],overdens:14,overrid:[1,14,18],page:8,pair:27,parallel:[19,20],paramet:[9,11,19,20,24,28],parent:[2,19,21],parsl:[],part:[19,20],particular:[26,27],pass:17,patch:26,pdf:[9,19,20,21,24],peopl:22,phot_data:11,photo:[9,19,20,21,24],photo_col:9,photo_data:9,photo_fil:9,photometr:[9,14,26],photometri:[9,11,14,16,19,20],photoz:25,photoz_mlz:25,photoz_stack:25,php:11,pickl:[26,27],pip:10,pipe:[],pipelin:[10,11,19,24,26,27],pipelinestag:[0,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27],pixel:[1,14,18,28],pixel_schem:[1,14,18,28],pixelizationschem:28,placehold:19,plain:[26,27],plot:[4,13,21,25,26],point:[3,19,20],point_estim:[19,20],posit:[11,24,26],potenti:[],power:[27,28],prepar:[14,19,20],prepare_mapp:[1,14,18],prepare_output:[19,20],present:22,process:[19,20],projecta:10,projectdir:10,psf:1,psf_diagnost:25,purer:[9,16],put:[2,11,24],python:[2,10],pz_data:24,pzpdfmlz:20,quantiti:24,r11:9,r12:9,r21:9,r22:9,r_std:9,random:[18,19,20],random_cat:25,randomli:[18,19],rang:[19,20],rank:[],raw:3,read:[9,11,14,24],read_nbin:26,real:3,redshift:[19,20,21],refer:[],region:26,reject:27,remove_undetect:9,repr:[],repres:27,represent:[26,27],requir:[8,14],resolut:28,respons:[9,11,24],retriev:9,rho:22,right:[2,14,19,20],robert:9,rogu:22,rotat:18,round:28,round_approx:28,row:[19,22],run:[3,10,11,14,19,20,21,24,26],sacc:3,sadli:22,same:[14,21,27],sampl:[11,24,26],save:[2,9,14,19,20],save_map:14,scheme:28,scipi:10,scp:10,sdss3:11,search:8,see:2,select:[11,14,24],select_len:11,selector:25,self:[26,27],separ:14,set:[9,10,11,21,24],setattr:[],setup:10,setup_output:[11,24],shape:[11,19,20,24],shear:[11,14,19,24,27],shear_catalog:27,shear_data:24,shear_tomography_catalog:24,should:[],signatur:[],similar:[19,20],simpler:24,simpli:2,simul:[3,9,16],singl:9,size:[9,17,24],snr:9,snr_limit:9,softwar:10,some:[10,14,19,20,24],sometim:[22,26],sourc:[0,1,2,3,4,5,6,7,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28],source_bin:24,source_selector:25,space:[10,21],special:2,specif:2,specifi:27,spectra:27,spit:[19,20],split:[21,24],src1:26,src2:26,stack:21,stage:[3,9,10,11,16,17,19,21,24,26,27],standard:[2,9,26],star:[22,26],start:[9,11,16,19,20,24],stat:22,statist:[19,20,22],step:[],stge:[],store:26,str:28,string:[26,27],strip:9,structur:14,subclass:[1,14,18,21,26,28],suit:18,suppli:[1,14,18],support:28,system:[],systemat:[26,27],tabl:[9,24],tag:2,take:26,target:11,target_s:9,teach:9,test:[2,9,16,26],thei:22,them:[11,22,24],theta:26,thi:[1,2,3,9,10,11,14,16,17,19,20,21,24,26,27],thing:[19,20],think:22,those:[11,24],through:[11,14,19,21,24],throughout:9,time:19,todo:[24,27],togeth:17,tomo_bin:[11,24],tomograph:[24,26,27],tomographi:[11,14,21,24],tomography_catalog:[11,27],tree:20,treecorr:26,trivial:3,truth:21,tupl:[26,27],two:[3,9,22,24],twopoint:25,twopoint_data_r:3,twopoint_data_real_raw:3,twopoint_fouri:25,txauxiliarymap:1,txbaselensselector:11,txbasemap:[1,14,18],txblind:3,txbrighterfatterplot:22,txbuzzardmock:9,txconvergencemap:4,txconvergencemapplot:4,txcosmodc2mock:9,txdensitymap:14,txdiagnosticplot:6,txexposureinfo:7,txexternallensmap:14,txexternallensnoisemap:18,txfouriergaussiancovari:5,txgammatbrightstar:26,txgammatdimstar:26,txgammatfieldcent:26,txgammatrandom:26,txingestredmag:0,txingeststar:16,txjackknifecent:26,txlensmap:14,txmainmap:14,txmapcorrel:12,txmapplot:13,txmeanlensselector:11,txmetacalgcrinput:16,txnoisemap:18,txnullblind:3,txphotozplot:21,txphotozsourcestack:21,txphotozstack:21,txpipe:[0,1,2,3,4,5,6,7,9,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28],txpsfdiagnost:22,txrandomcat:23,txrandomphotozpdf:19,txrealgaussiancovari:5,txrowestatist:22,txsimplemask:15,txsourcemap:14,txsourcenoisemap:18,txsourceselector:24,txsourcetruenumberdens:21,txstardensitytest:22,txstarsheartest:22,txtracermetadata:17,txtruenumberdens:21,txtruthlensselector:11,txtwopoint:26,txtwopointfouri:27,txtwopointlenscat:26,txtwopointplot:26,txtwopointplotsfouri:27,type:[2,19,20,28],unblind:3,under:[],undetect:9,unfortun:9,unit_respons:9,updat:14,use:[3,10,14,22,24,26],used:[9,11,16,19,20,24,28],useful:[9,16],user:[10,24],usernam:10,uses:[10,21,26],using:[10,26],usual:22,valu:[1,11,14,18,19,20,21,24,27,28],variant:[19,20,24],variou:10,vector:3,veri:28,version:2,want:[19,26,27],washington:9,weak:[],weight:14,when:10,where:[9,28],which:[2,10,14,19,20,24,26,27,28],who:22,whole:[19,20],wider:[],win:27,within:[],without:3,wonderfulli:22,workflow:[],would:24,wrapper:[2,26],write:[2,11,19,20,24],write_global_valu:[11,24],write_output:[9,19,20],write_tomographi:[11,24],www:11,you:10,zeljko:9,zuntz:10},titles:["Ingest redmagic","Auxilliary maps","Base stage","Blinding","Convergencemaps","covariance stage","Diagnostics plots","exposure_info","TXPipe documentation","Input Catalogs","TXPipe Installation","Lens selector","map_correlations","map_plots","maps","masks","metacal_gcr_input","metadata","noise_maps","photoz","photoz_mlz","photoz_stack","psf_diagnostics","random_cats","source_selector","Stages implemented in TX-Pipe:","Twopoint correlations","twopoint_fourier","Utilities"],titleterms:{auxilliari:1,base:2,blind:3,catalog:9,convergencemap:4,correl:26,covari:5,diagnost:6,document:8,exposure_info:7,implement:25,indic:8,ingest:0,input:9,instal:10,len:11,map:[1,14],map_correl:12,map_plot:13,mask:15,metacal_gcr_input:16,metadata:17,noise_map:18,photoz:19,photoz_mlz:20,photoz_stack:21,pipe:25,plot:6,psf_diagnost:22,random_cat:23,redmag:0,requir:10,selector:11,source_selector:24,stage:[2,5,25],tabl:8,twopoint:26,twopoint_fouri:27,txpipe:[8,10],util:28}})