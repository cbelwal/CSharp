using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public class SingleConvolutionLayerInput : LayerBase
    {       
        private int _depth; //_depth is same as number of filters
                     

        /// <summary>
        /// Parameterless constructor needed for use by derived classes
        /// </summary>
        public SingleConvolutionLayerInput()
        {  }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="data">3D array in form depth, columns, rows</param>
        public SingleConvolutionLayerInput(int inputDepth,
                                int inputColumns,
                                int inputRows)
        {
            _depth = inputDepth; //For Input Layer this will be mostly 3 for each RGB            
            SetupFilterUnits(_depth,inputColumns, inputRows);           
            //TODO: Add Padding in Input Layer
        }

        /// <summary>
        /// data[depth][column][rows]
        /// </summary>
        /// <param name="data"></param>
        public void SetData(double[][][] data)
        {
            //Verify
            if (data.Length != FilterUnits.Length)
                throw new InvalidDataException();

            if (data.Length > 0)
            {
                if(data[0].Length != FilterUnits[0].GetValueMapNoOfColumns() ||
                    data[0][0].Length != FilterUnits[0].GetValueMapNoOfRows())
                throw new InvalidDataException();
            } 
            //Assign Values
            for (int ii = 0; ii < FilterUnits.Length; ii++)
            {
                FilterUnits[ii].SetValueMap(data[ii]);
            }
        }

        private void SetupFilterUnits(int noFilterUnits,
                                      int noColumns,
                                      int noRows)
        {
            FilterUnits = new SingleFilterUnit[_depth];
            for (int ii = 0; ii < FilterUnits.Length; ii++)
                FilterUnits[ii] = new SingleFilterUnit( noColumns, noRows, 0, 0);
        }
        

    }
}
