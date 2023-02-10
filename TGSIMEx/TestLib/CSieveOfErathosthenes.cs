using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHPeriodLib
{
    public class CSieveOfEratosthenes : IPrimeNumberGenerator
    {
        private List<double> primeNumbers;
        private double[] primes;
        //private int noOfPrimes;

        public CSieveOfEratosthenes()
        {
            //noOfPrimes = 0;
        }

        public string Description
        {
            get { return "Sieve Of Eratosthenes Prime Number Generation Algorithm"; }
        }

        public string Developer
        {
            get { return "University of Houston"; }
        }

        public string Message
        {
            get { return ""; }
        }


        public List<double> GeneratePrimeNumbers(double min, 
                double max)
        {

            SieveOfEratosthenes(max);
            primeNumbers = new List<double>();

            foreach (double prime in primes)
                if (prime != 0 && prime >= min)
                    primeNumbers.Add(prime);


            return primeNumbers;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="max"></param>
        private void SieveOfEratosthenes(double max)
        {
            int ii, idx;
            double tmp;
            //noOfPrimes = 0;
            primes = new double[(int)max - 1];

            //Generate all Primes 
            idx = 0;
            
            for (ii = 2; ii <= max; ii++)
                primes[idx++] = ii;

            

            for (idx = 0; idx < max - 1; idx++)
            //for (idx = (int)max-2; idx >= 0; idx--)
            {
                tmp = 0;
                while (tmp == 0 && idx < max - 1) tmp = primes[idx++]; //Find next non zero prime
                //Remove multiples of tmp
                for (ii = idx + 1; ii < max - 1; ii++)
                //for (ii = (int)idx-1; ii >= 0; ii--)
                {
                    if (primes[ii] % tmp == 0) primes[ii] = 0;
                }
                if (Math.Pow(tmp, 2.0) >= max) break; //No need to go beyonds squares
            }

        }
    }
}
