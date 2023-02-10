using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Common.DataManagement
{ 
    public class DataSetCompact
    {
        private int _noOfColumns;
        private int _idxTargetValue;
        private int _maxParallelThreads;
        string[] _attributeHeaders;

        List<double[]> _allData;

        public DataSetCompact(string[] attributeHeaders,
                                   int idxTargetValue,
                                   int maxParallelThreads)
        {
            _attributeHeaders = attributeHeaders;
            _noOfColumns = attributeHeaders.Length;
            _idxTargetValue = idxTargetValue;
            _maxParallelThreads = maxParallelThreads;
            _allData = new List<double[]>();
        }

        public void AddSingleRow(double [] rowData)
        {
            if (rowData.Length > _noOfColumns)
                throw new InvalidDataException();
            _allData.Add(rowData);
        }

        /// <summary>
        /// Returns all data rows in a 2D array with column first order
        /// 
        /// The column first order is needed for all Build functions
        /// </summary>
        /// <returns></returns>
        public double[][] GetAllDataRows()
        {            
            double[][] allDataRows  =
                    ArrayManipulation.Get2DArray(_noOfColumns, _allData.Count, _maxParallelThreads);
            Parallel.For(0, _allData.Count, 
                        new ParallelOptions { MaxDegreeOfParallelism = _maxParallelThreads }, idxRow =>
            {
                for (int idxCol = 0; idxCol < _allData[0].Length; idxCol++)
                {
                    allDataRows[idxCol][idxRow] =
                            _allData[idxRow][idxCol];
                }                
            });
            return allDataRows;
        }

        public string[] GetAllAttributeHeaders()
        {
            return _attributeHeaders;
        }

        #region Single Value Getters
        public string GetSingleAttributeHeader(int idx)
        {
            return _attributeHeaders[idx];
        }

        public double[] GetSingleDataRowWithNoTargetValue(int rowIdx)
        {
            return ArrayManipulation.RemoveSpecificColumn1D(_allData[rowIdx], _idxTargetValue);
        }

        public double GetTargetValue(int rowIdx)
        {
            return _allData[rowIdx][_idxTargetValue];
        }

        public int GetIdxTargetAttribute()
        {
            return _idxTargetValue;
        }

        public int GetNumberOfDataRows()
        {
            return _allData.Count;
        }

        /// <summary>
        /// Formats a double value upto a specified decimals digit
        /// </summary>
        /// <param name="value"></param>
        /// <param name="decimals"></param>
        /// <returns></returns>
        public static string GetFormattedDouble(double value, int decimals)
        {
            return value.ToString($"F{decimals}");
        }

        #endregion Single Value Getters



    }
}
