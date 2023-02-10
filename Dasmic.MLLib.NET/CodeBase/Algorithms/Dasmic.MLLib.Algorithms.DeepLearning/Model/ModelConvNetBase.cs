using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Dasmic.MLLib.Algorithms.DeepLearning.Support;
using Dasmic.MLLib.Algorithms.NeuralNetwork.Support;

namespace Dasmic.MLLib.Algorithms.DeepLearning
{
    public class ModelConvNetBase :
        MLLib.Common.MLCore.ModelBase
    {
        private List<LayerBase> _layers { get; set; }
        private Dictionary<double, string> _targetValueMapping;

        public ConfigurationNN ConfigNN {get;set;} 

        public enum EnumMode
        {
            Regression,
            Classification
        }

        public EnumMode Mode;        
        
        

        /// <summary>
        /// Pass dummy values to base, make sure to call SetValues 
        /// to pass values
        /// </summary>
        public ModelConvNetBase(Dictionary<double, string> targetValueMapping) :base(0,0,0) //base contructor is called with dummy values
        {
            _targetValueMapping = targetValueMapping;
            Initialize();
        }

        private void Initialize()
        {
            Mode = EnumMode.Regression;
            _layers = new List<LayerBase>();
            ConfigNN = new ConfigurationNN();
        }
       
        //public override abstract double RunModelForSingleData(double[] data);

        /// <summary>
        /// Will return single value which is the maximum value
        /// among all output units
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public override double RunModelForSingleData(double[] data)
        {
            double[] computedValue = GetOutputValues(data);
            return 0;
            //Find max Idx in computedValue
            int maxIdx = computedValue.Select((item, indx) =>
                new { Item = item, Index = indx }).
                OrderByDescending(x => x.Item).Select(x => x.Index).First();

            if (Mode == EnumMode.Classification)
            {
                //return TargetValues[maxIdx];
            }
            else
                return computedValue[maxIdx];
        }

        /// <summary>
        /// Returns an array with output values of 
        /// all computed units
        /// </summary>
        /// <param name="data"></param>
        /// <returns></returns>
        public double[] GetOutputValues(double[] data)
        {
            VerifyDataForRun(data);
            return null;
        }

        public override void SaveModel(string filePath)
        {

        }

        //Deserialization Routine
        public override void LoadModel(string filePath)
        {

        }

        
        public void AddLayer(LayerBase layer)
        {
            //Set upstream layer to previous layer
            if(_layers.Count > 0) //Check if Input layer already added
                layer.SetUpstreamLayer(_layers[_layers.Count - 1]);

            _layers.Add(layer);       
        }

        /// <summary>
        /// Returns true if the last layer if the FC layer
        /// </summary>
        /// <returns></returns>
        public bool IsFCLayerLast()
        {
            if ((_layers[_layers.Count - 1]).GetType()
                    == typeof(SingleFullyConnectedLayer))
                return true;
            else
                return false;
        }

        public int GetNumberOfLayers()
        {
            return _layers.Count;
        }

        public LayerBase GetLayer(int idx)
        {
            return _layers[idx];
        }

    }
}
