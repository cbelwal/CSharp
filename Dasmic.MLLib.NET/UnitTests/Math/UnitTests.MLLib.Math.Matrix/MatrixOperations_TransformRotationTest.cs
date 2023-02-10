using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region Transpose
        [TestMethod]
        public void Matrix_transform_rotate_clockwise_2_row_step_1_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1,1);

            Assert.AreEqual(newMatrix[0][0], 2);
            Assert.AreEqual(newMatrix[0][1], 4);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 3);
        }


        [TestMethod]
        public void Matrix_transform_rotate_clockwise_2_row_step_3_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1, 3);

            Assert.AreEqual(newMatrix[0][0], 3);
            Assert.AreEqual(newMatrix[0][1], 1);
            Assert.AreEqual(newMatrix[1][0], 4);
            Assert.AreEqual(newMatrix[1][1], 2);
        }

        [TestMethod]
        public void Matrix_transform_rotate_3_row_step_1_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1, 1);

            Assert.AreEqual(newMatrix[0][0], 3);
            Assert.AreEqual(newMatrix[0][1], 6);
            Assert.AreEqual(newMatrix[0][2], 9);
            Assert.AreEqual(newMatrix[1][0], 2);
            Assert.AreEqual(newMatrix[1][1], 5);
            Assert.AreEqual(newMatrix[1][2], 8);
            Assert.AreEqual(newMatrix[2][0], 1);
            Assert.AreEqual(newMatrix[2][1], 4);
            Assert.AreEqual(newMatrix[2][2], 7);
        }

        [TestMethod]
        public void Matrix_transform_rotate_3_row_step_3_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1, 2);

            Assert.AreEqual(newMatrix[0][0], 9);
            Assert.AreEqual(newMatrix[0][1], 8);
            Assert.AreEqual(newMatrix[0][2], 7);
            Assert.AreEqual(newMatrix[1][0], 6);
            Assert.AreEqual(newMatrix[1][1], 5);
            Assert.AreEqual(newMatrix[1][2], 4);
            Assert.AreEqual(newMatrix[2][0], 3);
            Assert.AreEqual(newMatrix[2][1], 2);
            Assert.AreEqual(newMatrix[2][2], 1);
        }

        [TestMethod]
        public void Matrix_transform_rotate_3_row_step_4_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1, 4);

            Assert.AreEqual(newMatrix[0][0], 1);
            Assert.AreEqual(newMatrix[0][1], 2);
            Assert.AreEqual(newMatrix[0][2], 3);
            Assert.AreEqual(newMatrix[1][0], 4);
            Assert.AreEqual(newMatrix[1][1], 5);
            Assert.AreEqual(newMatrix[1][2], 6);
            Assert.AreEqual(newMatrix[2][0], 7);
            Assert.AreEqual(newMatrix[2][1], 8);
            Assert.AreEqual(newMatrix[2][2], 9);
        }

        
        [TestMethod]
        public void Matrix_transform_rotate_3_row_non_square_step_1_test()
        {
            InitData_dataset_3_rows_non_square();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1, 1);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 3);
            Assert.AreEqual(newMatrix[0][0], 2);
            Assert.AreEqual(newMatrix[0][1], 4);
            Assert.AreEqual(newMatrix[0][2], 6);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 3);
            Assert.AreEqual(newMatrix[1][2], 5);            
        }

        [TestMethod]
        public void Matrix_transform_rotate_counter_3_row_non_square_step_1_test()
        {
            InitData_dataset_3_rows_non_square();
            double[][] newMatrix =
                _mo.TransformRotateAntiClockwise(_matrix1, 3);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 3);
            Assert.AreEqual(newMatrix[0][0], 2);
            Assert.AreEqual(newMatrix[0][1], 4);
            Assert.AreEqual(newMatrix[0][2], 6);
            Assert.AreEqual(newMatrix[1][0], 1);
            Assert.AreEqual(newMatrix[1][1], 3);
            Assert.AreEqual(newMatrix[1][2], 5);
        }


        [TestMethod]
        public void Matrix_transform_rotate_counter_3_row_step_3_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newMatrix =
                _mo.TransformRotateAntiClockwise(_matrix1, 2);

            Assert.AreEqual(newMatrix[0][0], 9);
            Assert.AreEqual(newMatrix[0][1], 8);
            Assert.AreEqual(newMatrix[0][2], 7);
            Assert.AreEqual(newMatrix[1][0], 6);
            Assert.AreEqual(newMatrix[1][1], 5);
            Assert.AreEqual(newMatrix[1][2], 4);
            Assert.AreEqual(newMatrix[2][0], 3);
            Assert.AreEqual(newMatrix[2][1], 2);
            Assert.AreEqual(newMatrix[2][2], 1);
        }

        #endregion


        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixException))]
        public void Matrix_transform_rotate_invalid_data_throws_exception()
        {
            InitData_dataset_invalid();
            double[][] newMatrix =
                _mo.TransformRotateClockwise(_matrix1,1);
        }

    }
}
