using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    /// <summary>
    /// Intuitively the SingleLayer class used in Neutral Network
    /// can be used here. However, the upstreams layer for  the SingleLayer has
    /// linear arrays for weight and values which will not work correctly is used 'as-is'
    /// 
    /// This Layer is also different from Convolutional and Pooling layers
    /// since it does not use any instances of UnitBase. Here
    /// all weights for all units (depth) are defined in an array
    /// 
    /// Note that this layer assumes that the upstream layer ValueMap has only 1 value/unit
    /// Hence, a Pooling layer with filter size 1 has to be the upstream layer else error will be triggered
    /// </summary>
    public class SingleFullyConnectedLayer:LayerBase
    {
        private LayerBase _upStreamLayer; //Cannot use _upStreamLayer in base class
        private double[][] _weights; //_weights[idxUnit][idxUpunit]
        private double[] _errorTerm;     //Store error values
        private double[] _upStreamValues; //Persist upstream values for faster computations

        private IActivationFunction _activationFunction;

        int _noUpStreamUnits, _noUpStreamVMRows, _noUpStreamVMCols;
        //Upstream layer value map rows and cols        
        
        /// <summary>
        /// Upstream layer will generally be pooling layer
        /// 
        /// </summary>
        /// <param name="upStreamLayer"></param>
        /// <param name="uniqueTargetValues"></param>
        public SingleFullyConnectedLayer(LayerBase upStreamLayer,
                                            int uniqueTargetValues,
                                              int maxParallelThreads=-1)
        { 
            _upStreamLayer = upStreamLayer;
            _noOfUnits = uniqueTargetValues; //Same as depth
            _maxParallelThreads = maxParallelThreads;

            _noUpStreamUnits = _upstreamLayer.GetNumberOfUpstreamUnits();
            //Upstream layer value map rows and cols 
            _noUpStreamVMRows = _upstreamLayer.GetValueMapNoOfRows();
            _noUpStreamVMCols = _upstreamLayer.GetValueMapNoOfColumns();

            //Set Default Activation function
            _activationFunction = new Sigmoid();

            CheckUpstreamLayerUnits();
            //Call after all private vars have been assigned values
            InitializeWeights();
            
        }


        /// <summary>
        /// Check if the upstream layer units have Value Map of sizze 1x1
        /// 
        /// If not raise an exception
        /// </summary>
        private void CheckUpstreamLayerUnits()
        {
            if (_upstreamLayer.GetValueMapNoOfColumns() > 1 ||
                    _upstreamLayer.GetValueMapNoOfColumns() > 1)
                throw new NeuralNetworkConfigurationNotReady();                                             
        }

        private void SetActivationFunction(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }

        /// <summary>
        /// _weights[idxUnit][idxUpunit]
        /// </summary>
        private void InitializeWeights()
        {
            if (_weights == null) //Will not set if Weight is already assigned
            {
                _weights = new double[_noOfUnits][];                            
            }
          
            //for(int col=0;col< Weights.Length;col++)
            Parallel.For(0, _weights.Length,
                      new ParallelOptions
                      {
                          MaxDegreeOfParallelism =
                          _maxParallelThreads
                      }, unit =>
                      { //For each unit
                          _weights[unit] = new double[_noUpStreamUnits];
                          Random rnd = new Random();
                          int mul = 0;
                          for (int idxUpUnit = 0; idxUpUnit < _noUpStreamUnits; idxUpUnit++)
                          {
                              _weights[unit][idxUpUnit] = WeightBaseValue * mul;                              
                          } //idxUpUnit
                      });
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="idxUnit"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        public double GetValue(int idxUnit)
        {
            double net = 0;            
            Parallel.For(0, _noUpStreamUnits,
                      new ParallelOptions
                      {
                          MaxDegreeOfParallelism =
                          _maxParallelThreads
                      }, idxUpUnit =>
                      {
                          net += _weights[idxUnit][idxUpUnit] *
                                    getUpstreamValue(idxUpUnit);            
                        
                      });//idxUpUnit
            //Apply Activation function
            double value = _activationFunction.GetValue(net);
            return value;            
        }


        private double getUpstreamValue(int idxUpUnit)
        {
            return _upstreamLayer.GetValueMap(idxUpUnit, 0, 0); 
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="downStreamLayerError"></param>
        /// <param name="learningRate"></param>
        public void ComputeErrorAndUpdateWeights(double downStreamLayerError,
                                                           double learningRate, 
                                                           double [] expectedValue)
        {
            if (expectedValue.Length != _noOfUnits)
                throw new InvalidDataException();
            _errorTerm = new double[_noOfUnits];

            Parallel.For(0, _noOfUnits,
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxUnit =>
                    //for (int idxOutputUnit = 0; idxOutputUnit < computedValuesAllLayers[idxOutputLayer].Length; idxOutputUnit++)
                    {                        
                        _errorTerm[idxUnit] = ComputeErrorTerm(idxUnit, expectedValue[idxUnit]);
                        for (int idxUpUnit = 0; idxUpUnit < _noUpStreamUnits; idxUpUnit++)
                        {
                            double deltaWeight = learningRate *
                                            _errorTerm[idxUnit] *
                                            getUpstreamValue(idxUpUnit);

                            //Update the Weights
                            _weights[idxUnit][idxUpUnit] = _weights[idxUnit][idxUpUnit] + deltaWeight;
                        }
                    });//idxOutput            
        }

        /// <summary>
        /// Function should be called only during Backpropagation
        /// Error term should be computed first
        /// </summary>
        /// <param name="idxUnit"></param>
        /// <returns></returns>
        public double GetErrorTerm(int idxUnit)
        {
            return _errorTerm[idxUnit];
        }

        /// <summary>
        /// Compute the error term (error multiplied by derivatice)
        /// unit
        /// </summary>
        /// <param name="expectedValues">Actual value in regression, else 0 and 1 if incorrect value, should be equal to noOfUnits</param>
        /// <returns></returns>
        protected double ComputeErrorTerm(int idxUnit, 
                                            double expectedValue)
        {
            double computedValue = GetValue(idxUnit);
            double error = _activationFunction.GetDerivativeValue(computedValue) * (computedValue - expectedValue);
            return error;
        }

    }
}
