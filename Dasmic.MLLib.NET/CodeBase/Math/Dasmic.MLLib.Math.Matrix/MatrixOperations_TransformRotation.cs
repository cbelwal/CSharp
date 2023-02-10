using System;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        //This is for a general matrix
        //Row/Col concepts can differ from those used in trainingData
        /// <summary>
        /// Transform the matrix in such a way that it is rotated
        /// in a clockwise manner in 90 degree steps
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="deg90Steps"></param>
        /// <returns></returns>
        public double[][] TransformRotateClockwise(double[][]
                                matrix, int deg90Steps)
        {
            VerifyMatrix(matrix);

            int degSteps = deg90Steps % 4;

            double[][] newMatrix = matrix;
            for (int steps = 0; steps < degSteps; steps++)
            {
                //Transpose Matrix
                newMatrix = Transpose(newMatrix);
                double[][] tmpMatrix = new double[newMatrix.Length][];
                //Swap columns
                for(int col=0;col<newMatrix.Length/2;col++)
                {
                    tmpMatrix[col] = newMatrix[newMatrix.Length - 1 - col];
                    tmpMatrix[newMatrix.Length - 1 - col] = newMatrix[col];
                }
                //If odd columns copy center 
                if(newMatrix.Length % 2 != 0)
                {
                    tmpMatrix[newMatrix.Length/2] = 
                        newMatrix[newMatrix.Length/2];
                }

                newMatrix = tmpMatrix;
            }
            return newMatrix;
        }        
    


    public double[][] TransformRotateAntiClockwise(double[][]
                                matrix, int deg90Steps)
    {
        VerifyMatrix(matrix);
        //Use clockwise steps as Matrix column copy is less intensive operation
        int degClockwiseSteps = (4-deg90Steps) % 4;        
        return TransformRotateClockwise(matrix,degClockwiseSteps);
    }
}}
