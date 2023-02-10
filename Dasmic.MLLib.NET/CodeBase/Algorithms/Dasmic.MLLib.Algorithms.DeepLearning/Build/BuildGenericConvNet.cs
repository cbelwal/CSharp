using System.Threading.Tasks;
using System.Collections.Generic;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;
using Dasmic.MLLib.Algorithms.DeepLearning.Support;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    /// <summary>
    /// Generic Structure: INPUT - CONV - RELU - POOL - FC
    /// </summary>
    public class BuildGenericConvNet:BuildConvNetBase
    {        

        //Pooling
        private int _sizeWindow;
        private int _strideWindow;       

        //noOfUnits 
        #region Private variables to store state
        SingleConvolutionLayerInput _inputLayer;
        
        #endregion Private variables to store state

     
        /// <summary>
        /// Add layers from left to right
        /// 
        /// Input layer is added automatically
        /// </summary>
        /// <param name="newLayer"></param>
        public void AddLayer(LayerBase newLayer)
        {
            //Number of Filters and Filder Size should be specified in each Layer
            _model.AddLayer(newLayer);
        }

        
        /// <summary>
        /// Initialize model by passing target values of training data
        /// </summary>
        /// <param name="inputDepth"></param>
        /// <param name="inputColumns"></param>
        /// <param name="inputRows"></param>
        /// <param name="padding"></param>
        /// <param name="targetValueMapping">Key/Value pair of target values. This allows both double and string values to be stored</param>
        public BuildGenericConvNet(int inputDepth, 
                                int inputColumns, 
                                int inputRows,                                
                                int numberDataSamples,
                                Dictionary<double, string> targetValueMapping):base(inputDepth, inputColumns, 
                                                                                        inputRows, numberDataSamples,targetValueMapping)
        {
                     
        }

        
        /// <summary>
        /// Call the AddLayer function before calling this function
        /// to build the model
        /// 
        /// </summary>
        /// <param name="trainingData">3D array having a single training data with depth in 3rd Dimension. 
        /// Indexed as [depth][columns][rows]</param> 
        /// 
        /// This function will have to called for each training data
        /// 
        /// Call InitModel3D before calling this
        /// 
        /// This different approach has been adopted since training data could be very huge to be accomodated in memory
        /// This way only one training data needs to be present in memory at any given time
        /// <param name="indexTargetAttribute"></param>
        /// <returns>boolean value is stopping condition reached ot not</returns>
        public override bool
           BuildModelSingle(double[][][] trainingData,
                                double targetValue)
        {
            bool stoppingConditionReached = false;
            
            //Check if FC Layer is last
            if (!_model.IsFCLayerLast())
                throw new LastLayerNotFullyConnectedLayer();

            //Set single instance/row of training data
            _inputLayer.SetData(trainingData);

            /* ----------------- CAN BE DELE TED-----------------------
            //Assume all          
            //_model.Mode = _mode; //Classification or Regression 
                                 //Find out number of categories
                                 //Get unique value in Target Attribute
            //_model.TargetValues =
            //    GetNumberOfTargetValues(_mode, trainingData, indexTargetAttribute);
            //In Classification number of units same as number of categories
            //_noOfUnitsOutputLayer = _model.TargetValues.Length;
            ------------------------------- */

            if (VerifyLayers())
                throw new NeuralNetworkConfigurationNotReady();

            //Initialize weights and setup Upstream Layer for Hidden Layers
            //Dont parallelize as InitWeigts is already Parallel
            //for (int idxLayer = 1; idxLayer < noOfHiddenLayers + 1; idxLayer++)
            //{
            //    _model.GetLayer(idxLayer).SetUpstreamLayer(_model.GetLayer(idxLayer - 1));
            //    _model.GetLayer(idxLayer).InitializeWeights();
            //}
            
            double learningRate = 0;

            //Start the training
            //Do not parallelize as computation has to be sequential
            //double[] computedValuesOutput = new double[_model.GetOutputUnitCount()];
            double[][] computedValuesAllLayers = new double[_model.GetNumberOfLayers()][];
            double[][] errorLayers = new double[_model.GetNumberOfLayers()][];

            //Initialize Layer specific data
            //for(int idxLayer=0;idxLayer< errorLayers.Length;idxLayer++)
            /*
            Parallel.For(0, errorLayers.Length,
                            new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                            idxLayer =>
                            {
                                errorLayers[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)]; //Input layer is not included in original index
                                computedValuesAllLayers[idxLayer] = new double[_model.GetNumberOfUnits(idxLayer)];
                            });
*/

            int idxOutputLayer = _model.GetNumberOfLayers() - 1;
            //Start iterations
            //for (int epoch = 0; epoch < _noOfEpoch; epoch++) //Do not parallelize
            //{

            //learningRate = _alpha;// * (1.0 - (epoch /_noOfEpoch));/R also keeps learning rate same                

            //for (int row = 0; row < _noOfDataSamples; row++) //For each data sample
            //{
            /*
                _totalThreshold += ComputeError(computedValuesAllLayers, 
                                                    errorLayers, idxOutputLayer, row);
                //Fill input layer values in computedValuesAllLayers
                for (int idxUnit = 0; idxUnit < _model.GetNumberOfUnits(0); idxUnit++)
                {
                    computedValuesAllLayers[0][idxUnit] = //CAUTION: Upstream value is dependent on weights
                            _model.GetUnitValue(0, idxUnit, row);
                }

            //Update weights batch on batch value
            //If batchValue = 1 this workd like SGD
            if (_currentDataSampleRow++ % Configuration.BatchCount==0) 
                    UpdateWeights(computedValuesAllLayers, errorLayers, learningRate, row);
        //} //For row                    

        //Check Error Threshold
        if (_currentDataSampleRow++ % _noOfDataSamples == 0)//Compare when end of row
        {
            _currentDataSampleRow = 0;
            if (totalThreshold < Configuration.ErrorThreshold)
                stoppingConditionReached = true;//Stopping condition reached
                                                //  }//epoch
        }

        if (++_epoch >= Configuration.MaxEpoch)
            stoppingConditionReached = true;
        */
            return stoppingConditionReached;
        }


        /// <summary>
        /// Returns combined error value
        /// </summary>
        /// <param name="computedValuesAllLayers"></param>
        /// <param name="errorLayers"></param>
        /// <param name="idxOutputLayer"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        protected double ComputeError(double[][] computedValuesAllLayers,
                                    double[][] errorLayers,
                                    int idxOutputLayer, int row)
        {
            return 0;
            /*double sumError = 0;
            object mutex = new object();
            //For each output unit, can go in parallel - Compute errors for output unit 
            Parallel.For(0, computedValuesAllLayers[idxOutputLayer].Length,
                    new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                    idxOutputUnit =>
                    //for (int idxOutputUnit = 0; idxOutputUnit < computedValuesAllLayers[idxOutputLayer].Length; idxOutputUnit++)
                    {
                        double expectedOutput = 0;

                        computedValuesAllLayers[idxOutputLayer][idxOutputUnit] =
                                        _model.GetOutput(idxOutputUnit, row);

                        //Will depend on mode
                        //Number of units in outputs =  model.TargetValues
                        if (_mode == ModelBackPropagationBase.EnumMode.Classification) //Classification
                        {
                            //expectedOutput = 1 if correct output unit
                            //expectedOutput = 0 if incorrect output unit
                            if (_trainingData[_indexTargetAttribute][row]
                                        == _model.TargetValues[idxOutputUnit])
                                expectedOutput = 1;
                            else
                                expectedOutput = 0;
                        }
                        else //for regression
                            expectedOutput = _trainingData[_indexTargetAttribute][row];


                        errorLayers[idxOutputLayer][idxOutputUnit] = _model.GetDerivativeValue(idxOutputLayer,
                                                    computedValuesAllLayers[idxOutputLayer][idxOutputUnit]) *
                                                    GetOutputUnitDifference(computedValuesAllLayers[idxOutputLayer][idxOutputUnit], expectedOutput);
                        lock (mutex)
                        {
                            sumError += Math.Abs(errorLayers[idxOutputLayer][idxOutputUnit]);
                        }
                    });//idxOutput

            //Compute errors for all hidden unit 
            //Start from right most  to left most
            for (int idxHiddenLayer = idxOutputLayer - 1; idxHiddenLayer > 0; idxHiddenLayer--)
            {
                Parallel.For(0, computedValuesAllLayers[idxHiddenLayer].Length,
                        new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                        idxHiddenUnit =>
                        //for (int idxHiddenUnit = 0; idxHiddenUnit < computedValuesAllLayers[idxHiddenLayer].Length;
                        //                           idxHiddenUnit++)
                        {
                            computedValuesAllLayers[idxHiddenLayer][idxHiddenUnit] =
                                _model.GetOutputTillLayer(idxHiddenLayer, idxHiddenUnit, row);

                            double sumHidden = 0;
                            //Compute sum of output unit
                            for (int idxUnitRightLayer = 0; idxUnitRightLayer < computedValuesAllLayers[idxHiddenLayer + 1].Length; idxUnitRightLayer++)
                            {
                                sumHidden +=
                                           errorLayers[idxHiddenLayer + 1][idxUnitRightLayer] * //Use error of output layer
                                               _model.GetWeight(idxHiddenLayer + 1, idxHiddenUnit, idxUnitRightLayer); //+2 is the actual downstream layer idx
                            }

                            errorLayers[idxHiddenLayer][idxHiddenUnit] =
                                 _model.GetDerivativeValue(idxHiddenLayer,
                                                            computedValuesAllLayers[idxHiddenLayer][idxHiddenUnit]) *
                                                            (sumHidden);
                        });
            } //idxHiddenLayer
            return sumError;*/
        }

        #region Verification Functions
        /// <summary>
        /// This function can be called separate to verify if layers 
        /// are good
        /// </summary>
        /// <returns></returns>
        public bool VerifyLayers()
        {
            bool exceptionFlag = false;
            Parallel.For(0, _model.GetNumberOfLayers() - 1,
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   idxLayer =>
                   {
                       if (_model.GetLayer(idxLayer) == null)
                           exceptionFlag = true;
                   });
            return exceptionFlag & VerifyUpstreamLayers();
        }

        protected bool VerifyUpstreamLayers()
        {
            bool exceptionFlag = false;
            Parallel.For(1, _model.GetNumberOfLayers(),
                   new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads },
                   idxLayer =>
                   {
                       if (_model.GetLayer(idxLayer).GetNumberOfUpstreamUnits() < 0)//Check if Upstream Layer is setup, this will set an exception otherwise
                           exceptionFlag = true;
                   });
            return exceptionFlag;
        }
        #endregion Verification Functions    
    }
}
