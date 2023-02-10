using Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SinglePoolingUnit : UnitBase
    {
        private IPoolingFunction _poolingFunction;
        private int _poolSize;
        private int _idxUpUnit;//Index of the unit in upstream layer this pooling unit is connected to. One pooling unit only connects to one conv unit (1 pooling unit for each depth in upstream conv. layer)

        public SinglePoolingUnit(LayerBase upStreamLayer,                                        
                                        int windowSize,
                                        int strideSize,
                                        int idxUpUnit) : base(upStreamLayer, strideSize,windowSize)
        {
            _poolSize = windowSize;
            _idxUpUnit = idxUpUnit;
            SetValueMap(); //Sets the Output Values 2D array
        }

        public void SetPoolingFunction(IPoolingFunction activationFunction)
        {
            _poolingFunction = activationFunction;
        }

        /// <summary>
        /// Computes the ValueMap using the provided Pooling function
        /// </summary>       
        public override void ComputeValueMap()
        {
            //padding has to be less than or equal to window size
            int poolSize = _poolSize;

            if (_padding > poolSize)
                throw new InvalidPaddingValueException();
            if (_poolingFunction == null)
                throw new InvalidPoolingFunctionException();
           

            for (int idxRowVM = 0; idxRowVM < GetValueMapNoOfRows(); idxRowVM++)
            {//idxRowVM
                for (int idxColVM = 0; idxColVM < GetValueMapNoOfColumns(); idxColVM++)
                { //idxColVM                    
                    //int idxUpValueMapCol=0, idxUpValueMapRow=0;
                    //Set the filter window co-ordinates in the upstream value map
                    //relative to the current Value Map                                                           
                    _valueMap[idxColVM][idxRowVM] = ComputeValueMapForIndex(idxColVM, idxRowVM,
                                                                   _idxUpUnit);
                    //idxFilter                    
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
                                            int idxUpUnit)
        {
            int idxUpValueMapCol = 0, idxUpValueMapRow = 0;
            
            SingleFilterUnit upFilterUnit =
                  (SingleFilterUnit)_upStreamLayer.GetFilterUnit(idxUpUnit);
            int upVMNoOfCols = upFilterUnit.GetValueMapNoOfColumns();
            int upVMNoOfRows = upFilterUnit.GetValueMapNoOfRows();

            //padding will always be 0
            int upMapLeftCol = idxColVM * _stride - _padding;
            int upMapLeftRow = idxRowVM * _stride - _padding;
            double value = 0;
            _poolingFunction.Reset();
            //TODO: ALSO INVERT FILTER FOR CROSS CORELATION
            for (int idxFilterRow = 0; idxFilterRow < _poolSize; idxFilterRow++)
            {
                for (int idxFilterCol = 0; idxFilterCol < _poolSize; idxFilterCol++)
                {
                    //Update value                               
                    idxUpValueMapCol = upMapLeftCol + idxFilterCol;
                    idxUpValueMapRow = upMapLeftRow + idxFilterRow;
                    //Multiply upStreamFilterMap by current Filter weights
                    if (idxUpValueMapCol >= 0 && idxUpValueMapRow >= 0 &&
                            idxUpValueMapCol < upVMNoOfCols && idxUpValueMapRow < upVMNoOfRows)
                    { //else value is 0 since it is coming from zero padding
                        _poolingFunction.AddValue(idxUpValueMapCol,idxUpValueMapRow,
                                                    upFilterUnit.GetValueMapAtIndex(idxUpValueMapCol,
                                                                    idxUpValueMapRow));
                    }
                } //idxFilterCol

            } //idxFilterRow

            value = _poolingFunction.GetValue();
            return value;
        }
    }
}
