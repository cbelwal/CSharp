using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction
{
    public class AveragePooling : IPoolingFunction
    {
        List<double> _values;
        
        public AveragePooling()
        {
            _values = new List<double>();
        }
        
        /// <summary>
        /// Returns the max values among values
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public double GetValue()
        {
            double sum=0;
            foreach (double d in _values)
            {
                sum += d;
            }
            return sum / _values.Count;
        }

        /// <summary>
        /// Index is not useful for Avg. Pooling
        /// </summary>
        /// <returns></returns>
        public int GetIndexCol()
        {
            return 0;
        }

        /// <summary>
        /// Index is not useful for Avg. Pooling
        /// </summary>
        /// <returns></returns>
        public int GetIndexRow()
        {
            return 0;
        }

        public void AddValue(int indexCol,int indexRow, 
                                double value)
        {
            _values.Add(value);
        }

        public void Reset()
        {
            //TODO
            _values.Clear();
        }
    }
}
