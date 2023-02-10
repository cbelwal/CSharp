using System;
using System.Collections.Generic;
using System.Collections;
using System.Text;

namespace TGSIMExLib
{
    public static class CMath
    {
        /// <summary>
        /// Computes the Lowest Common Multiple of numbers inside NumberList
        /// </summary>
        /// <param name="numberList"></param>
        /// <returns></returns>
        public static double LCM(List<double> numberList)
        {
            long ii, jj;
            double [] factors;
            double maxFactor, divisor, LCM = 0;
            ArrayList divisors = new ArrayList();

            factors = new double[numberList.Count];
            ii = 0;
            maxFactor = 0;
            
            //Copy factors
            foreach (double f in numberList)
            {
                factors[ii++] = f;
                if (maxFactor < f) maxFactor = f;
            }

            divisor = 0;
            for (ii = 2; ii <= maxFactor; ii++)
            {

                if (divisor != 0)
                {
                    divisors.Add(divisor);
                    ii = (long)divisor;
                }
                divisor = 0;
                for (jj = 0; jj < factors.Length; jj++)
                {
                    if (factors[jj] % ii == 0) //Perfectly divisible
                    {
                        factors[jj] = factors[jj] / ii;
                        divisor = ii;
                    }
                }
            }

            //Add last divisor
            if (divisor != 0)
                divisors.Add(divisor);


            //Now compute LCM
            LCM = 1;
            foreach (double d in divisors)
                LCM = LCM * d;

            return LCM;
        }



        /// <summary>
        /// Computes the factorial of a discrete number
        /// </summary>
        /// <param name="number"></param>
        /// <returns></returns>
        public static long Factorial(long lNumber, long lStopAtNumber)
        {
           long ii;
            long fact = 1;

            if (lStopAtNumber < 1 || lStopAtNumber > lNumber) lStopAtNumber = 1;

            for (ii = lNumber; ii > lStopAtNumber; ii--)
                fact = fact * ii;

            return fact;
        }
    }
}
