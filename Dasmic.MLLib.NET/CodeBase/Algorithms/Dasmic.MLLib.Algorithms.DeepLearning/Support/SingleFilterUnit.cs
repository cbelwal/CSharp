using System;
using System.Threading.Tasks;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Math.Matrix;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    /// <summary>
    /// In CNN each Filter is like a Unit in a single layer
    /// 
    /// In the standard NN we did not implement a class for each Unit
    /// However, to better separate the complexity in CNNs later's we have created
    /// a separate class for each Unit (also called as Filter in CNN) here.
    /// </summary>
    public class SingleFilterUnit:UnitBase
    {
        private double[][][] _weights; //[depth][col][row]
        private double _bias; //Keep bias separate
        private double _weightBaseValue;
        

        IActivationFunction _activationFunction;

        /// <summary>
        /// Overloaded constructor. Mainly meant to be called from Input layer
        /// </summary>
        /// <param name="noOfInputColumns"></param>
        /// <param name="noOfInputRows"></param>
        /// <param name="windowSize"></param>
        /// <param name="strideSize"></param>
        /// <param name="maxParallelThreads"></param>
        /// <param name="weightBaseValue"></param>
        public SingleFilterUnit(int noOfInputColumns,
                                int noOfInputRows,
                                  int windowSize,  //FilterSize is always square
                                  int strideSize) : base(null, strideSize,
                                                                    windowSize)
        {
            SetValueMap(noOfInputColumns, noOfInputRows);
            SetActivationFunction(new RectifiedLinearUnit());
        }

        public SingleFilterUnit(LayerBase upStreamLayer,
                                  int windowSize,  //FilterSize is always square
                                  int strideSize) :base(upStreamLayer,strideSize,
                                                                     windowSize)
        {
            //The weight matrix should equal FilterSize                         
            _weightBaseValue = .005;
            InitializeWeightValues(upStreamLayer.GetNumberOfUnits(), windowSize);                        
            SetValueMap();            
            SetActivationFunction(new RectifiedLinearUnit());
        }

        /// <summary>
        /// Sets the padding manually. Other default padding value
        /// based on size of window is set
        /// </summary>
        public void SetPadding(int padding)
        {
            _padding = padding;
            SetValueMap();
        }

        public void SetWeightBaseValue(double value)
        {
            _weightBaseValue = value;
            //_weight vector is already set
            InitializeWeightValues(_weights.Length, 
                                        _weights[0].Length);
        }


        /// <summary>
        /// Function will overwrite assigned value 
        /// which is set from the Constructor
        /// </summary>
        /// <param name="idxUnit"></param>
        /// <param name="idxCol"></param>
        /// <param name="idxRow"></param>
        /// <param name="singleValue"></param>
        public void SetWeightValue(int idxUnit, int idxCol, int idxRow, 
                                        double singleValue)
        {            
            _weights[idxUnit][idxCol][idxRow] = singleValue;            
        }

        public void SetAllWeightsSingleValue(double singleValue)
        {
            InitializeWeightValues(_weights.Length,
                                        _weights[0].Length,singleValue);
        }

        public void SetActivationFunction(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;
        }

       

        /// <summary>
        /// Initialize weights for in filterUnit
        /// </summary>
        /// <param name="initValue"></param>
        private void InitializeWeightValues(int noOfUpUnits, int windowSize,
                                            double assignValue=Double.MaxValue )
        {
            _weights = new double[noOfUpUnits][][]; //noOfUpUnits is same as Depth
            
            Parallel.For(0, noOfUpUnits,
                      new ParallelOptions
                     {
                          MaxDegreeOfParallelism =
                          _maxParallelThreads
                      }, idxUpUnit =>
                      //for(int idxUpUnit=0;idxUpUnit < noOfUpUnits;idxUpUnit++)
                      {
                          _weights[idxUpUnit] = SupportFunctions.Get2DArray(windowSize,
                                            windowSize);
                          Random rnd = new Random();
                          int mul = 0;

                          for (int col = 0; col < _weights[idxUpUnit].Length;col++)
                              for (int row = 0; row < _weights[idxUpUnit][col].Length; row++)
                          {
                              //Use different init values
                              if (assignValue == Double.MaxValue)
                              {
                                  mul = rnd.Next(1, 10);
                                  _weights[idxUpUnit][col][row] = _weightBaseValue *
                                                            mul; //.005
                              }
                              else
                              {
                                  _weights[idxUpUnit][col][row] = assignValue;
                              }
                        }
                      });
            if (assignValue == Double.MaxValue)
                _bias = _weightBaseValue; //Bias initial value, can also multiply by random number
            else
                _bias = assignValue;
        }

        /// <summary>
        /// Computes the ValueMap(FilterMap) by convolution over each Input array
        /// from Upstream layer
        /// 
        /// This function contains the sliding convolution window logic
        /// </summary>
        /// <param name="upStreamLayer"></param>
        public override void ComputeValueMap()
        {
            //padding has to be less than or equal to window size
            int filterSize = _weights[0].Length;

            if (_padding > filterSize)
                throw new InvalidPaddingValueException();
            if (_activationFunction == null)
                throw new InvalidActivationFunctionException();

            double[][][] rotatedWeights = GetRotatedWeights();

            for (int idxRowVM = 0; idxRowVM < GetValueMapNoOfRows(); idxRowVM++)
            {//idxRowVM
                for (int idxColVM = 0; idxColVM < GetValueMapNoOfColumns(); idxColVM++)
                { //idxColVM
                    double sum = 0;
                    //int idxUpValueMapCol=0, idxUpValueMapRow=0;
                    //Set the filter window co-ordinates in the upstream value map
                    //relative to the current Value Map                                       
                    for (int idxUpUnit = 0; idxUpUnit < _upStreamLayer.GetNumberOfUnits();
                                    idxUpUnit++) //For each Filter Unit Upstream
                                                 //Filter is same as window
                                                 //Slide the filter Window to compute value for each cell in the ValueMap (VM)                                            
                    {                        
                        sum += ComputeValueMapForIndex(idxColVM, idxRowVM,
                                                        rotatedWeights, idxUpUnit);
                    } //idxFilter
                    //Normalize Sum
                    sum += _bias * 1.0; //Do for bias term
                    sum = sum / (filterSize * 2); //divide sum by filter Grid size
                    //Apply activation function - this saves time if applied here
                    sum = _activationFunction.GetValue(sum);
                    _valueMap[idxColVM][idxRowVM] = sum; //Assign value to FilterMap
                } //idxColFM
            } //idxRowFM  

        }


        /// <summary>
        /// Computes value of single filter map after accounting for padding
        /// </summary>
        /// <param name="idxColumn"></param>
        /// <param name="idxRow"></param>
        /// <returns></returns>
        private double ComputeValueMapForIndex(int idxColVM, 
                                            int idxRowVM,
                                            double [][][] rotatedWeights,
                                            int idxUpUnit)
        {
            int idxUpValueMapCol = 0, idxUpValueMapRow = 0;
            int filterSize = _weights[0].Length; //Filter is same as window
                                              //Set the filter window co-ordinates in the upstream value map
                                              //relative to the current Value Map
            SingleFilterUnit upFilterUnit =
                  (SingleFilterUnit)_upStreamLayer.GetFilterUnit(idxUpUnit);
            int upVMNoOfCols = upFilterUnit.GetValueMapNoOfColumns();
            int upVMNoOfRows = upFilterUnit.GetValueMapNoOfRows();

            //TODO: Fix this
            int upMapLeftCol = idxColVM * _stride - _padding;
            int upMapLeftRow = idxRowVM * _stride - _padding;          
            double sum=0;
            
            //TODO: ALSO INVERT FILTER FOR CROSS CORELATION
            for (int idxFilterRow = 0; idxFilterRow < filterSize; idxFilterRow++)
            {
                for (int idxFilterCol = 0; idxFilterCol < filterSize; idxFilterCol++)
                {
                    //Update value                               
                    idxUpValueMapCol = upMapLeftCol + idxFilterCol;
                    idxUpValueMapRow = upMapLeftRow + idxFilterRow;
                    //Multiply upStreamFilterMap by current Filter weights
                    if (idxUpValueMapCol >= 0 && idxUpValueMapRow >= 0 &&
                            idxUpValueMapCol < upVMNoOfCols && idxUpValueMapRow <upVMNoOfRows)
                    { //else value is 0 since it is coming from zero padding
                        sum += rotatedWeights[idxUpUnit][idxFilterCol][idxFilterRow] *
                                    upFilterUnit.GetValueMapAtIndex(idxUpValueMapCol,
                                                        idxUpValueMapRow);                                
                    }
                } //idxFilterCol
                 
            } //idxFilterRow
            
            return sum;
        }
        

        /// <summary>
        /// Convolution operation is same as co-relation, provided 
        /// weights are rotated by 180 deg
        /// </summary>
        /// <returns>180 degree rotated matrix</returns>
        public double[][][] GetRotatedWeights()
        {
            MatrixOperations mo = new MatrixOperations();
            double[][][] rotatedWeights = new double[_weights.Length][][];
            Parallel.For(0, _weights.Length,
                     new ParallelOptions
                     {
                         MaxDegreeOfParallelism =
                         _maxParallelThreads
                     }, idxUpUnit =>
                     {
                         rotatedWeights[idxUpUnit] =
                                mo.TransformRotateClockwise(_weights[idxUpUnit], 2);
                     });

            return rotatedWeights;
        }


        public double GetWeight(int idxUpUnit, int idxCol, int idxRow)
        {
            return _weights[idxUpUnit][idxCol][idxRow];
        }

        public void SetWeight(int idxUpUnit, int idxCol, 
                                    int idxRow,double value)
        {
            _weights[idxUpUnit][idxCol][idxRow] = value;
        }

        public override void ComputeErrorAndUpdateWeights(double downStreamLayerError)
        {
            throw new NotImplementedException();
        }


    }


}

