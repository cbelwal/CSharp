 namespace Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction
{
    public interface IPoolingFunction
    {
        double GetValue();
        int GetIndexRow();
        int GetIndexCol();
        void AddValue(int colIndex, int rowIndex,double value);
        void Reset();
    }
}
