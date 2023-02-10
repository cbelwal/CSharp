

namespace Dasmic.MLLib.Algorithms.NeuralNetwork.Support
{
    /// <summary>
    /// Contains configuration paramters of Neural Net of 
    /// </summary>
    public class ConfigurationNN
    {
        /// <summary>
        /// Set default values in constructor
        /// </summary>
        public ConfigurationNN()
        {
            MaxEpoch = 1000;
            Alpha = .3;            
            WeightBaseValue = .005;
            ErrorThreshold = .01;
            BatchCount = 1;
        }
        
        public int MaxEpoch {get;set;}
        public double Alpha { get; set; }
        public double WeightBaseValue { get; set; }
        public double ErrorThreshold { get; set; }
        public double BatchCount { get; set; } //If BatchCount=1 it is SGD

    }
}
