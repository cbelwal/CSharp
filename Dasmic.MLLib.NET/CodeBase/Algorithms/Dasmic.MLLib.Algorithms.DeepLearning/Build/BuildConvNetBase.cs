using System;
using System.Collections.Generic;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Algorithms.DeepLearning.Support;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    public class BuildConvNetBase : MLLib.Common.MLCore.BuildBase
    {
        #region Properties
        public ConfigurationNN Configuration;
        #endregion Properties
        #region Computation related properties
        protected int _epoch;
        protected ModelConvNetBase _model;
        protected SingleConvolutionLayerInput _inputLayer;
        protected double _learningRate;
        protected int _currentDataSampleRow;
        protected double totalThreshold;
        #endregion Computation related properties
        //Default is ReLU
        protected IActivationFunction ActivationFunctionConvolution { get; set; }

        //Convolution
        public int NoOfFeatures { get; set; }
        public int SizeOfFeatures { get; set; }
        public int StrideConvolution { get; set; }
        


        public BuildConvNetBase(int inputDepth,
                               int inputColumns,
                               int inputRows,                              
                               int numberDataSamples,
                               Dictionary<double, string> targetValueMapping)
        {
            _model = new ModelConvNetBase(targetValueMapping);
            Configuration = new ConfigurationNN();
            _noOfDataSamples = numberDataSamples;
            StrideConvolution = 2; //Default assignment
            //Add the input layer here
            _inputLayer = new SingleConvolutionLayerInput(inputDepth, inputColumns, inputRows);
            _model.Mode = ModelConvNetBase.EnumMode.Classification;
            _model.AddLayer(_inputLayer);
            _currentDataSampleRow = 0;
            _learningRate = Configuration.Alpha;
        }

        /// <summary>
        /// Sets either Classification of Regressison modes
        /// </summary>
        public ModelConvNetBase.EnumMode Mode
        {
            get
            {
                return _model.Mode;
            }
            set
            {
                _model.Mode = value;
            }
        }
    

        public ModelConvNetBase GetModel()
        {
            return _model;
        }
        

        public override Common.MLCore.ModelBase
          BuildModel(double[][] trainingData,
                       string[] attributeHeaders,
                       int indexTargetAttribute)
        {
            throw new NotImplementedException();
        }        

        public override bool BuildModelSingle(double[][][] trainingData,
                   double targetValue) //Mapping between target values and their string values
        {
            throw new NotImplementedException();
        }
       
        public override void InitializeModel(int inputDepth,
                                int inputColumns,
                                int inputRows,
                                int padding,
                                Dictionary<double, string> targetValueMapping)
        {
            throw new NotImplementedException();
        }
    }
}
