using System;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public abstract class UnitBase
    {
        protected int _stride;
        protected int _maxParallelThreads;
        protected int _padding;
        protected int _windowSize;
        protected LayerBase _upStreamLayer;

        //Known as ComputedValue in other NN        
        //Known as FilterMap in a Convolutional layer
        protected double[][] _valueMap;  //Will be used by both Convolution and Pooling layers

        protected void SetValueMap()
        {
            int noOfInputColumns = _upStreamLayer.GetValueMapNoOfColumns();
            int noOfInputRows = _upStreamLayer.GetValueMapNoOfRows();
            SetValueMap(noOfInputColumns, noOfInputRows);
        }

        /// <summary>
        /// Set the size of the Value map and initialize it
        /// </summary>
        /// <param name="noOfInputColumns"></param>
        /// <param name="noOfInputRows"></param>
        /// <param name="windowSize"></param>
        protected void SetValueMap(int noOfInputColumns, 
                                    int noOfInputRows)
        {            
            //Computed value will depend on stride size
            int noOfOutputCols = noOfInputColumns;
            int noOfOutputRows = noOfInputRows;

            //For Input Layer ComputedValue/FilterMap values remain the same
            if (_windowSize != 0) //If not an input layer
            {
                //Standard formula to determine FilterMap size
                double _temp = ((double)(noOfInputColumns - _windowSize + 2*_padding)
                                        / (double)_stride) + 1;
                if (System.Math.Round(_temp) != _temp)
                    throw new InvalidPaddingValueException(); 
                else
                    noOfOutputCols = Convert.ToInt32(_temp);

                _temp = ((double)(noOfInputRows - _windowSize + 2*_padding)
                                       / (double)_stride) + 1;
                if (System.Math.Round(_temp) != _temp)
                    throw new InvalidPaddingValueException();
                else
                    noOfOutputRows = Convert.ToInt32(_temp);
            }
            //Set FilterMap Size            
            _valueMap = SupportFunctions.Get2DArray(noOfOutputCols,
                                                        noOfOutputRows);
        }

     

        public void SetMaxParallelThreads(int value)
        {
            _maxParallelThreads = value;
        }

        public UnitBase(LayerBase upStreamLayer,
                            int stride,    
                            int windowSize)
        {
            _stride = stride;
            _maxParallelThreads =-1;
            _windowSize = windowSize;
            _upStreamLayer = upStreamLayer;
            _padding = 0;            
        }

        public int GetValueMapNoOfRows()
        {
            return _valueMap[0].Length;
        }

        public  int GetValueMapNoOfColumns()
        {
            return _valueMap.Length;
        }

        public double GetValueMapAtIndex(int idxCol, int idxRow)
        {
            return _valueMap[idxCol][idxRow];
        }

        public abstract void ComputeValueMap();

        /// <summary>
        /// Only use in Input Layer where input 
        /// needs to be directly copied
        /// </summary>
        public void SetValueMap(double[][] data)
        {
            _valueMap = data;
        }

        public virtual void ComputeErrorAndUpdateWeights(double downStreamError)
        {
            throw new NotImplementedException();
        }

    }
}
