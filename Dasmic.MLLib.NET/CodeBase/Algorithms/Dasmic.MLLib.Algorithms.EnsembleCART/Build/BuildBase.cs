using Dasmic.MLLib.Common.MLCore;

namespace Dasmic.MLLib.Algorithms.EnsembleCART
{
    public abstract class BuildBase : MLLib.Common.MLCore.BuildBase
    {
        protected int _numberOfTrees;

        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                             string[] attributeHeaders,
                             int indexTargetAttribute);
 
    
    }
}
