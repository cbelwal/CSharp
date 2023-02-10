using Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SingleConvolutionLayer: SingleConvolutionLayerInput
    {
        private IActivationFunction _activationFunction;
      
        public SingleConvolutionLayer(  int noOfFilterUnits,
                                        int sizeWindow, 
                                        int sizeStride)
        {
            _noOfUnits = noOfFilterUnits;
            _sizeWindow = sizeWindow;
            _sizeStride = sizeStride;
            _activationFunction = new RectifiedLinearUnit();
            WeightBaseValue = .05;            
        }

        public override void SetUpstreamLayer(LayerBase upStreamLayer)
        {
            _upstreamLayer = upStreamLayer;
            //Call this function only when upstream layer is known
            SetupFilterUnits(_noOfUnits,
                                _sizeWindow, //This also specifies the Filter dimensions
                                _sizeStride);
        }

        /// <summary>
        /// Default activation function is ReLU.Use this function to
        /// overwrite the default value
        /// </summary>
        /// <param name="activationFunction"></param>
        public void SetActivationFunction(IActivationFunction activationFunction)
        {
            _activationFunction = activationFunction;

            //Update activation in each filter
            foreach(SingleFilterUnit sfn in FilterUnits)
            {
                sfn.SetActivationFunction(_activationFunction);
            }
        }
        
        /// <summary>
        /// Call this function when upStreamLayer is known
        /// </summary>
        /// <param name="noOfFilterUnits"></param>
        /// <param name="sizeWindow"></param>
        /// <param name="sizeStride"></param>
        private void SetupFilterUnits(int noOfFilterUnits,
                                        int sizeWindow, 
                                        int sizeStride)
        {
            FilterUnits = new SingleFilterUnit[noOfFilterUnits];

            for(int ii=0;ii<FilterUnits.Length;ii++)
            {
                FilterUnits[ii] = new SingleFilterUnit(
                                        _upstreamLayer,                                        
                                        sizeWindow, 
                                        sizeStride);                                        
            }                                    
        }

        public void ComputeValueOfFilters()
        {
            //For each filter neuron in Layer
            foreach (SingleFilterUnit sfn in FilterUnits)
            {
                //Each FilterUnit has fix output
                //Each filter neuron is connected to another Filter Nruron in Upstream Layer
                sfn.ComputeValueMap();
            }
        }

    }
}
