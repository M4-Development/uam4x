##############################################################################
#
# PLEASE SEE UQ FRAMEWORK USER'S MANUAL FOR INFORMATION REGARDING THE INPUTS.
#
##############################################################################

#    Framework Inputs
FrameworkInputsDict={
    'FrameworkInputFolder' :r'E:\NASA_UAM\m4.uncertainty\m4\uncertainty\Framework\installer\Examples\1\InputFolder',
    'FrameworkOutputFolder':r'E:\NASA_UAM\m4.uncertainty\m4\uncertainty\Framework\installer\Examples\1\OutputFolder',
    'LoggingDebug'         :True
    }

#    User-specified System Information
SystemInputsDict={
    'SystemName'           :'Beam_Deflection_1',
    'SystemDirectory'      :r'E:\NASA_UAM\m4.uncertainty\m4\uncertainty\Framework\installer\Examples\1',
    'SystemHeadCommand'    :'',
    'SystemFilename'       :'Example_1.py',
    'SystemTailCommand'    :'',
    'SystemInputFolder'    :'',
    'SystemInputFile'      :'',
    'SystemInputTemplate'  :'',
    'SystemOutputFolder'   :'',
    'SystemOutputFile'     :'',
    'ImportPython'         :True,
    'LoadSamples'          :False,
    'SamplesFilename'      :'mysamples.dat',
    'LoadOutputs'          :False,
    'OutputFilename'       :'myoutput.dat',
    'LoadSensitivities'    :False,
    'SensitivitiesFilename':'mysensitivities.dat',
    'SystemFunction'       :'beamModel',
    'SystemMultiProcessing':(False,3),
    }

#    Uncertainty Quantification Method Information
UQInputsDict={
    'UQAnalysisType'       :'Aleatory',
    'UQAnalysisMode'       :'NIPC',
    'NIPCMethod'           :'Regression',
    'UseSensitivities'     :False,
    'Optimization'         :False,
    'QMUOptions'           :(False,'Lower'),
    'QMUPerformanceFiles'  :['QMU_LowerPerformance_0.dat','QMU_DesignPerformance_0.dat','QMU_UpperPerformance_0.dat'],
    'ProbabilityLevels'    :[0.025,0.25,0.5,0.75,0.975],
    'DiffOptions'          :('Forward',10e-6),
    'SensitivityReporting' :(90.0,1),
    'SamplingMethod'       :'LHS',
    'SamplingSize'         :1000,
    'ElimRandomVariables'  :[],
    'ElimRVNominalValues'  :{},
    'ComputeCorrelation'   :True,
    'MultiProcessing'      :(False,2),
    'HDF5'                 :False,
    'WriteArraysToFile'    :True,
    'NumRandomVariables'   :2,
    'RVNames'              :['Bending Moment (M)','Yield Stress (Y)'],
    'RVDistribution'       :[1,2],
    'RVDistValues'         :{(0,1):(2500.,200.),
                             (1,2):(2000.,2800.)},
    'PolynomialOrder'      :5,
    'OverSamplingRatio'    :2,
    'NumOutputVariables'   :1,
    'OutputVarNames'       :['Normalized Beam Deflection'],
    'ComputeSobolIndices'  :True,
    'GeneratePDFCDFs'      :True,
    'PDFCDFSamplingMethod' :'LHS',
    'PDFCDFNumSamples'     :5000,
    'ConfidenceInterval'   :95.0,
    'SOP_SamplingMethod'   :'LHS',
    'SOP_InnerSamples'     :1000,
    'SOP_OuterSamples'     :1000,
    'SOP_PrintCDF'         :10,
    'SOP_HDF5'             :False,
    'SOP_LoadSamples'      :False,
    'SOP_InnerSamplesFile' :'sopAsamps.dat',
    'SOP_OuterSamplesFile' :'sopEsamps.dat',
    'LoadPDFCDFSamples':(False,'X'),
    'PDFCDFSamplesFile':'',
    }
