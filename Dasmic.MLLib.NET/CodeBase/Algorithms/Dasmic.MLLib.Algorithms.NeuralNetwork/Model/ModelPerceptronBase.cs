using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Algorithms.NeuralNetwork
{
    public abstract class ModelPerceptronBase : ModelBase
    {
        public ModelPerceptronBase(double missingValue,
                        int indexTargetAttribute,
                        int countAttributes) :
                                base(missingValue, indexTargetAttribute, countAttributes)
        {

        }


        public double GetWeightOutputLayer(int idxUpUnit, 
                                    int idxUnit)
        {
            return GetWeight(1,idxUpUnit, idxUnit);
        }

        public void SetWeightOutputLayer(int idxUpUnit,
                                            int idxUnit, double value)
        {
            SetWeight(1,idxUpUnit, idxUnit, value);
        }

        public double GetBiasOutputLayer(int idxUnit)
        {
            return GetWeight(1, GetNumberOfUpstreamUnits(1) - 1, idxUnit);
                                
        }

        public void SetBiasOutputLayer(int idxUnit,double value)
        {
            SetWeight(1, GetNumberOfUpstreamUnits(1),idxUnit,  value);
        }

        public override abstract double RunModelForSingleData(double[] data);

    }
}
