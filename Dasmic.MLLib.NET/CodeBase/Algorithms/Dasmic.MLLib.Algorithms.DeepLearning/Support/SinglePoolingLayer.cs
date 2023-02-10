using Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SinglePoolingLayer:LayerBase
    {
        private IPoolingFunction _poolingFunction;

        public SinglePoolingLayer( int sizeWindow, //Both Height and Width will be same
                                       int sizeStride, 
                                       int padding)
        {
            _sizeWindow = sizeWindow;
            _sizeStride = sizeStride;
            _poolingFunction = new MaxPooling();
            
        }

        public override void SetUpstreamLayer(LayerBase upStreamLayer)
        {
            _upstreamLayer = upStreamLayer;
            //auto determine number of units
            _noOfUnits = _upstreamLayer.GetNumberOfUnits(); //No of pooling Units should equal upstream Filter neurons

            //Call this function only when upstream layer is known
            SetupPoolingUnits(_noOfUnits,
                                _sizeWindow, //This also specifies the Filter dimensions
                                _sizeStride);
        }

        /// <summary>
        /// Call only when upStream units are available
        /// </summary>
        /// <param name="noOfPoolingUnits">Will be same as number of upstream units</param>
        /// <param name="sizeFilter"></param>
        /// <param name="sizeStride"></param>
        private void SetupPoolingUnits(int noOfPoolingUnits,
                                        int sizeFilter,
                                        int sizeStride)
        {
            FilterUnits = new SinglePoolingUnit[noOfPoolingUnits];
            for (int ii = 0; ii < FilterUnits.Length; ii++)
            {
                FilterUnits[ii] = new SinglePoolingUnit(
                                        _upstreamLayer,
                                        sizeFilter,
                                        sizeStride,ii);
            }
        }

        /// <summary>
        /// Default pooling function is MaxPool. Use this function to
        /// overwrite the default value
        /// </summary>
        /// <param name="poolingFunction"></param>
        public void SetPoolingFunction(IPoolingFunction poolingFunction)
        {
            _poolingFunction = poolingFunction;

            //Update activation in each filter
            foreach (SinglePoolingUnit spn in FilterUnits)
            {
                spn.SetPoolingFunction(_poolingFunction);
            }
        }
    }
}
