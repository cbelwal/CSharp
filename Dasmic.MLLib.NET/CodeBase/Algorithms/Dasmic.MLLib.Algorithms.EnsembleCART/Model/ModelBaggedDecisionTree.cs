
namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public class ModelBaggedDecisionTree : ModelBase
    {
        public ModelBaggedDecisionTree(double missingValue,
                          int indexTargetAttribute, 
                          int countAttributes, int noOfTrees) :
                                base(missingValue, 
                                    indexTargetAttribute, countAttributes, noOfTrees)
        {

        }


    }
}
