using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Algorithms.DecisionTree;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public class ModelRandomForest : ModelBase
    {
        public ModelRandomForest(double missingValue,
                   int indexTargetAttribute, int countAttributes, int noOfTrees) :
                                base(missingValue, 
                                    indexTargetAttribute, countAttributes,noOfTrees)
        {

        }

        public void AddTree(int idxTree, ModelCART tree, int[] index)
        {
            _allTrees[idxTree]=tree;
        }

    }
}
