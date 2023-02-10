using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Algorithms.DeepLearning.Support;
using Dasmic.MLLib.Algorithms.NeuralNetwork;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.DeepLearning
{
    [TestClass]
    public class SingleFilterUnitTest:BaseTest
    {

        #region Depth 1
        #region Stride 1
        [TestMethod]
        public void compute_value_map_3_by_3_padding_1_stride_1_depth_1()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli = 
                                new SingleConvolutionLayerInput(1,3,3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 1);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(1);            
            sfu.ComputeValueMap();


            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 4);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 4);
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(0, 0), 2.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(0, 1), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(0, 2), 6.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(0, 3), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(1, 0), 6.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(1, 1), 13.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(1, 2), 17.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(1, 3), 10.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(2, 0), 12.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(2, 1), 25.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(2, 2), 29.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(2, 3), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(3, 0), 8.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(3, 1), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(3, 2), 18.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(3, 3), 10.0 / 4.0));
        }

        [TestMethod]
        public void compute_value_map_3_by_3_padding_2_stride_1_depth_1()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 1);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(2);
            sfu.ComputeValueMap();

            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 6);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 6);
            
            //Col 0
            int col = 0;            
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));            
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 1
            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 2.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 6.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 2
            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 6.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 13.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 17.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 10.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 3
            col = 3;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 12.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 25.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 29.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 4
            col = 4;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 8.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 18.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 10.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 5
            col = 5;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void compute_value_map_3_by_3_padding_3_stride_1_depth_1_expects_exception()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 1);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(3);
            sfu.ComputeValueMap();
        }
        #endregion Stride 1

        #region Stride 2
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void compute_value_map_3_by_3_padding_1_stride_2_depth_1()
        {
            set_data_depth_1_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(1);
            sfu.ComputeValueMap();            
        }

        [TestMethod]
        public void compute_value_map_4_by_4_padding_1_stride_2_depth_1()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(1);
            sfu.ComputeValueMap();


            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 3);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 3);
            
            //Col 0
            int col = 0;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 2.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 6.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 5.0 / 4.0));            
            
            //Col 1
            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 15.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 35.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 21.0 / 4.0));

            //Col 2
            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 14.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 30.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 17.0 / 4.0));
            
        }

        [TestMethod]
        public void compute_value_map_4_by_4_padding_2_stride_2_depth_1()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(2);
            sfu.ComputeValueMap();

            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 4);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 4);

            //Col 0
            int col = 0;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));

            //Col 1
            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 15.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 23.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));

            //Col 2
            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 47.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 55.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));

            //Col 3
            col = 3;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));
        }

        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void compute_value_map_4_by_4_padding_3_stride_2_depth_1_expects_exception()
        {
            set_data_depth_1_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(1, 4, 4);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(3);
            sfu.ComputeValueMap();
        }


        #endregion Stride 2
        #endregion Depth 1

        #region  Depth 3
        #region Stride 1
        [TestMethod]
        public void compute_value_map_3_by_3_padding_1_stride_1_depth_3()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 1);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(1);
            sfu.ComputeValueMap();


            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 4);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 4);

            int col = 0;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 10.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 10.0 / 4.0));

            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 37.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 49.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 28.0 / 4.0));

            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 34.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 73.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 85.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 46.0 / 4.0));

            col = 3;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 22.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 46.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 52.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 28.0 / 4.0));
        }

        
        [TestMethod]
        public void compute_value_map_3_by_3_padding_2_stride_1_depth_3()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 1);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(2);
            sfu.ComputeValueMap();

            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 6);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 6);

            //Col 0
            int col = 0;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 1
            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 10.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 10.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 2
            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 37.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 49.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 28.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 3
            col = 3;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 34.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 73.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 85.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 46.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 4
            col = 4;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 22.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 46.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 52.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 28.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));

            //Col 5
            col = 5;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 4), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 5), 1.0 / 4.0));
        }

        
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void compute_value_map_3_by_3_padding_3_stride_1_depth_3_expects_exception()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 1);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(3);
            sfu.ComputeValueMap();
        }
        #endregion Stride 1

        
        #region Stride 2
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void compute_value_map_3_by_3_padding_1_stride_2_depth_3()
        {
            set_data_depth_3_cols_3();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 3, 3);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(1);
            sfu.ComputeValueMap();
        }

        
        [TestMethod]
        public void compute_value_map_4_by_4_padding_1_stride_2_depth_3()
        {
            set_data_depth_3_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 4, 4);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(1);
            sfu.ComputeValueMap();


            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 3);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 3);

            //Col 0
            int col = 0;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 4.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 16.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 13.0 / 4.0));

            //Col 1
            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 43.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 103.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 61.0 / 4.0));

            //Col 2
            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 40.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 88.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 49.0 / 4.0));

        }

        
        [TestMethod]
        public void compute_value_map_4_by_4_padding_2_stride_2_depth_3()
        {
            set_data_depth_3_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 4, 4);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(2);
            sfu.ComputeValueMap();

            Assert.AreEqual(sfu.GetValueMapNoOfColumns(), 4);
            Assert.AreEqual(sfu.GetValueMapNoOfRows(), 4);

            //Col 0
            int col = 0;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));

            //Col 1
            col = 1;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 43.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 67.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));

            //Col 2
            col = 2;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 139.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 163.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));

            //Col 3
            col = 3;
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 0), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 1), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 2), 1.0 / 4.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(sfu.GetValueMapAtIndex(col, 3), 1.0 / 4.0));
        }

        
        [TestMethod]
        [ExpectedException(typeof(InvalidPaddingValueException))]
        public void compute_value_map_4_by_4_padding_3_stride_2_depth_3_expects_exception()
        {
            set_data_depth_3_cols_4();
            SingleConvolutionLayerInput scli =
                                new SingleConvolutionLayerInput(3, 4, 4);
            scli.SetData(_trainingData3D);
            SingleFilterUnit sfu = new SingleFilterUnit(scli, 2, 2);
            sfu.SetAllWeightsSingleValue(1); //All filter values are 1
            sfu.SetPadding(3);
            sfu.ComputeValueMap();
        }
        #endregion Stride 2
        #endregion Depth 2
    }
}
