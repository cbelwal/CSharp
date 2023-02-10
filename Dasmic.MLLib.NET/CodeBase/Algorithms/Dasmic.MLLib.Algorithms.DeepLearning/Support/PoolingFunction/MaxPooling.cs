using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction
{
    public class MaxPooling: IPoolingFunction
    {        
        private double _maxValue;
        private int _indexRow;
        private int _indexCol;

        public MaxPooling()
        {
            Reset();
        }


        /// <summary>
        /// Returns the max values among values
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double GetValue()
        {
            return _maxValue;
        }

        public int GetIndexCol()
        {
            return _indexCol;
        }

        public int GetIndexRow()
        {
            return _indexRow;
        }

        public void Reset()
        {
            _maxValue = double.MinValue;
            _indexCol = int.MaxValue;
            _indexRow = int.MaxValue;
        }

        public void AddValue(int indexCol, int indexRow, double value)
        {
            if (value > _maxValue)
            {
                _maxValue = value;
                _indexCol = indexCol;
                _indexRow = indexRow;
            }
        }
    }
}
