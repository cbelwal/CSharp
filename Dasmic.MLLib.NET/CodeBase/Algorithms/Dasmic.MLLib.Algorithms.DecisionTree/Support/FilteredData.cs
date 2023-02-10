using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace Dasmic.MLLib.Algorithms.DecisionTree
{
    public class FilteredData
    {
        private double [][] _filteredData;
        private ConcurrentBag<int> _trainingDataRowIndices;
        private int _numberOfRows;

        public FilteredData(double[][] filteredData,
                                ConcurrentBag<int> trainingDataRowIndices,
                                int numberOfRows)
        {
            _filteredData = filteredData;
            _trainingDataRowIndices = trainingDataRowIndices;
            _numberOfRows = numberOfRows;
        }

        public double[][] FilteredDataValues
        {
            get
            {
                return _filteredData;
            }
        }

        public ConcurrentBag<int> TrainingDataRowIndices
        {
            get
            {
                return _trainingDataRowIndices;
            }
        }

        public int NumberOfRows
        {
            get
            {
                return _numberOfRows;
            }
        }


    }
}
