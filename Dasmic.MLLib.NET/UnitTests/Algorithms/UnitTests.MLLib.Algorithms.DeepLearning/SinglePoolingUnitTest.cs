using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DeepLearning.Support;
using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;
using Dasmic.MLLib.Algorithms.DeepLearning.Support.PoolingFunction;

namespace UnitTests.MLLib.Algorithms.DeepLearning
{
    [TestClass]
    public class SinglePoolingUnitTest : BaseTest
    {
        #region Depth 1       
        [TestMethod]
        public void pool_value_map_3_by_3_stride_1_window_2_depth_1_maxpool()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 1,0);           
            spu.SetPoolingFunction(new MaxPooling());            
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 5.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 8.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 9.0));            
        }

        [TestMethod]
        public void pool_value_map_3_by_3_stride_1_window_2_depth_1_avgpool()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 1, 0);           
            spu.SetPoolingFunction(new AveragePooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 3.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 7.0));
        }

        [TestMethod]
        public void pool_value_map_3_by_3_stride_1_window_3_depth_1_maxpool()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 3, 1, 0);           
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 1);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 1);
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 9.0));
            
        }

        
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void pool_value_map_3_by_3_stride_2_window_2_depth_1_maxpool_exception()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 2, 0);          
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();         
        }


        [TestMethod]
        public void pool_value_map_4_by_4_stride_1_window_2_depth_1_maxpool()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 1, 0);           
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 3);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 3);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 7.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 2), 8.0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 10.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 11.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 2), 12.0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(2, 0), 14.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(2, 1), 15.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(2, 2), 16.0));


        }

        
        [TestMethod]
        public void pool_value_map_4_by_4_stride_1_window_2_depth_1_avgpool()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 1, 0);            
            spu.SetPoolingFunction(new AveragePooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 3);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 3);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 14.0/4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 18.0/4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 2), 22.0/4.0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 30.0/4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 34.0/4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 2), 38.0/4.0));

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(2, 0), 46.0/4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(2, 1), 50.0/4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(2, 2), 54.0/4.0));
        }

        [TestMethod]
        public void pool_value_map_4_by_4_stride_1_window_3_depth_1_maxpool()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 3, 1, 0);            
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 11.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 12.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 15.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 16.0));
        }

        [TestMethod]
        public void pool_value_map_4_by_4_stride_2_window_2_depth_1_maxpool()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 2, 0);
            //spu.SetAllWeightsSingleValue(1); //All filter values are 1
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 8.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 14.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 16.0));
        }

        [TestMethod]
        public void pool_value_map_4_by_4_stride_1_window_4_depth_1_maxpool()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 4, 1, 0);            
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 1);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 1);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 16.0));            
        }

        
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void pool_value_map_4_by_4_stride_3_window_2_depth_1_maxpool_exception()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 3, 0);           
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();
        }

        #endregion Depth 1
        
            
        #region Depth 3       
        [TestMethod]
        public void pool_value_map_3_by_3_stride_1_windows_2_depth_3_maxpool()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3,3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 1, 2);            
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 5.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 8.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 9.0));
        }
        
        [TestMethod]
        public void pool_value_map_3_by_3_stride_1_window_2_depth_3_avgpool()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 1, 2);            
            spu.SetPoolingFunction(new AveragePooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 3.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 7.0));
        }

        
        [TestMethod]
        public void pool_value_map_3_by_3_stride_1_windows_3_depth_3_maxpool()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 3, 1, 1);
           
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 1);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 1);
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 9.0));
        }
        
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void pool_value_map_3_by_3_stride_2_depth_3_maxpool_exception()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 2, 1);
            
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();
        }

        [TestMethod]
        public void pool_value_map_4_by_4_stride_2_window_2_depth_3_maxpool()
        {
            set_data_depth_3_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 2, 1);
            //spu.SetAllWeightsSingleValue(1); //All filter values are 1
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 2);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 2);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 6.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 1), 8.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 0), 14.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(1, 1), 16.0));
        }

        
        [TestMethod]
        public void pool_value_map_4_by_4_stride_1_window_4_depth_3_maxpool()
        {
            set_data_depth_3_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 4, 1, 2);
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();

            Assert.AreEqual(spu.GetValueMapNoOfColumns(), 1);
            Assert.AreEqual(spu.GetValueMapNoOfRows(), 1);

            Assert.IsTrue(SupportFunctions.DoubleCompare(spu.GetValueMapAtIndex(0, 0), 16.0));
        }

        
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void pool_value_map_4_by_4_stride_3_window_2_depth_3_maxpool_exception()
        {
            set_data_depth_3_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 4, 4);
            scli.SetData(_trainingData3D);
            SinglePoolingUnit spu = new SinglePoolingUnit(scli, 2, 3, 2);
            spu.SetPoolingFunction(new MaxPooling());
            spu.ComputeValueMap();
        }
        
        #endregion Depth 3

    }
}
