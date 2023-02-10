using Dasmic.MLLib.UnitTest.Core;

namespace UnitTests.MLLib.Algorithms.DeepLearning
{
    public class BaseTest : UnitTestBase
    {
        protected double[][][] _trainingData3D;

        protected void set_data_depth_1_cols_3()
        {
            _trainingData3D = new double[1][][];
            _trainingData3D[0] = new double[3][];
            _trainingData3D[0][0] = new double[] { 1, 2, 3 };
            _trainingData3D[0][1] = new double[] { 4, 5, 6 };
            _trainingData3D[0][2] = new double[] { 7, 8, 9 };
        }

        protected void set_data_depth_1_cols_4()
        {
            _trainingData3D = new double[1][][];
            _trainingData3D[0] = new double[4][];
            _trainingData3D[0][0] = new double[] { 1, 2, 3, 4 };
            _trainingData3D[0][1] = new double[] { 5, 6, 7, 8 };
            _trainingData3D[0][2] = new double[] { 9, 10, 11, 12 };
            _trainingData3D[0][3] = new double[] { 13, 14, 15, 16 };
        }


        protected void set_data_depth_3_cols_4()
        {
            _trainingData3D = new double[3][][];
            _trainingData3D[0] = new double[4][];
            _trainingData3D[1] = new double[4][];
            _trainingData3D[2] = new double[4][];            

            _trainingData3D[0][0] = new double[] { 1, 2, 3, 4 };
            _trainingData3D[0][1] = new double[] { 5, 6, 7, 8 };
            _trainingData3D[0][2] = new double[] { 9, 10, 11, 12 };
            _trainingData3D[0][3] = new double[] { 13, 14, 15, 16 };

            _trainingData3D[1][0] = new double[] { 1, 2, 3, 4 };
            _trainingData3D[1][1] = new double[] { 5, 6, 7, 8 };
            _trainingData3D[1][2] = new double[] { 9, 10, 11, 12 };
            _trainingData3D[1][3] = new double[] { 13, 14, 15, 16 };

            _trainingData3D[2][0] = new double[] { 1, 2, 3, 4 };
            _trainingData3D[2][1] = new double[] { 5, 6, 7, 8 };
            _trainingData3D[2][2] = new double[] { 9, 10, 11, 12 };
            _trainingData3D[2][3] = new double[] { 13, 14, 15, 16 };            
        }

        protected void set_data_depth_3_cols_3()
        {
            _trainingData3D = new double[3][][];
            _trainingData3D[0] = new double[3][];
            _trainingData3D[1] = new double[3][];
            _trainingData3D[2] = new double[3][];

            _trainingData3D[0][0] = new double[] { 1, 2, 3 };
            _trainingData3D[0][1] = new double[] { 4, 5, 6 };
            _trainingData3D[0][2] = new double[] { 7, 8, 9 };

            _trainingData3D[1][0] = new double[] { 1, 2, 3 };
            _trainingData3D[1][1] = new double[] { 4, 5, 6 };
            _trainingData3D[1][2] = new double[] { 7, 8, 9 };

            _trainingData3D[2][0] = new double[] { 1, 2, 3 };
            _trainingData3D[2][1] = new double[] { 4, 5, 6 };
            _trainingData3D[2][2] = new double[] { 7, 8, 9 };

        }

        /// <summary>
        //Initialize data from Jason's book
        /// R:
        /// X = list(c(1, 2, 4, 3, 5))
        /// Y = list(c(1, 3, 3, 2, 5))
        /// </summary>
        protected void Init_dataset_jason_linear_regression()
        {
            _attributeHeaders = new string[] {
                                     "X",
                                     "Y"};
            _indexTargetAttribute = 1;

            _trainingData = new double[2][];
            _trainingData[0] = new double[] { 1, 2, 4, 3, 5 };
            _trainingData[1] = new double[] { 1, 3, 3, 2, 5 };
        }

        /// <summary>
        /// Load the Pythagoras Data set
        /// </summary>
        protected void Init_dataset_pythagoras()
        {
            LoadFromDataSet(EnumDataSets.Pythagoras, -1);
        }
    }
}
