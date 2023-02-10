using System;

namespace Dasmic.MLLib.Algorithms.DeepLearning.Support
{
    public abstract class LayerBase
    {
        protected LayerBase _upstreamLayer;
        protected int _maxParallelThreads;
        protected int _noOfUnits; //Filter Units in case of Conv. layer
        protected int _sizeWindow;
        protected int _sizeStride;
        protected int _padding;

        protected UnitBase[] FilterUnits;

        #region Properties
        public int MaxParallelThreads { get; set; }
        public double WeightBaseValue { get; set; }
        #endregion Properties


        public LayerBase()
        {
            MaxParallelThreads = -1;
            WeightBaseValue = .005; //default
            _padding = 0;
        }
        

        /// <summary>
        /// Returns number of  Units
        /// 
        /// NOTE: Does not incluse bias term
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfUnits()
        {
            return FilterUnits.Length;
        }

        public UnitBase GetFilterUnit(int idx)
        {
            return FilterUnits[idx];
        }

        /// <summary>
        /// Does not include bias term
        /// </summary>
        /// <returns></returns>
        public int GetNumberOfUpstreamUnits()
        {
            return _upstreamLayer.GetNumberOfUnits();
        }

        /// <summary>
        /// All units will have the same ValueMap size
        /// </summary>
        /// <returns></returns>
        public int GetValueMapNoOfColumns()
        {
            return FilterUnits[0].GetValueMapNoOfColumns();
        }
          
        public int GetValueMapNoOfRows()
        {
            return FilterUnits[0].GetValueMapNoOfRows();
        }

        public double GetValueMap(int idxUnit,int idxCol, int idxRow)
        {
            return FilterUnits[idxUnit].GetValueMapAtIndex(idxCol, idxRow);
        }

        public virtual void SetUpstreamLayer(LayerBase upStreamLayer)
        {
            throw new NotImplementedException();
        }

        public virtual void ComputeErrorAndUpdateWeights(double downStreamLayerError,
                                                           double learningRate )
        {
            throw new NotImplementedException();
        }

    }
}
