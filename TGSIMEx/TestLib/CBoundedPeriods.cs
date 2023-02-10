using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHPeriodLib
{
    public class CBoundedPeriods:IPeriodGenerator
    {

        private List<double> _primes;
        private List<CPeriodSet> _allPeriods;
        private List<double> [] _groups;
        private string _Message;

        public string Message
        {
            get { return _Message; }
        }



        /// <summary>
        /// 
        /// </summary>
        /// <param name="oConfiguration"></param>
        /// <returns></returns>
        public List<CPeriodSet> GetPeriods(CConfiguration oConfiguration, 
                    IPrimeNumberGenerator primeGenerator)
        {
            
            int maxGroups;
            int maxIdx;
            int idx = 0;
            double maxFactor;
            List<double> tmpList;
            int ii;
            double delta;
            double maxPrime;

            delta = oConfiguration.MaxPeriod - oConfiguration.MinPeriod + 1;
            maxPrime = Math.Ceiling(delta / 2);

            _allPeriods = new List<CPeriodSet>();
            //Generate Primes
            if (_primes == null) GeneratePrimes(oConfiguration);

            maxGroups = GetMaximumGroups(oConfiguration.MinPeriod,oConfiguration.MaxPeriod,
                            oConfiguration.NumberOfTasksPerSet);
            _Message = "";

            _groups = new List<double>[maxGroups];

            maxFactor = GetMaximumFactor(oConfiguration.MinPeriod,oConfiguration.MaxPeriod,
                            oConfiguration.NumberOfTasksPerSet);


            maxIdx = GetIndexOfMaxPrime(maxFactor);

            
            //Fill in Periods in All Groups
            //Starting with Highest Index First
            idx = 0;
            int primeIdx =  maxIdx;
            while (primeIdx >= 0)
            {
                tmpList = GeneratePeriods(oConfiguration.MinPeriod,
                                            oConfiguration.MaxPeriod, _primes[primeIdx--]);

                _groups[idx++] = tmpList;
            }

            //Now Return Periods
            CCombinationGenerator oComb = new CCombinationGenerator();
            CPeriodSet ps;

            //Customize the group number here
            for (ii = 0; ii < _groups.Length; ii++)
            {
                tmpList = _groups[ii];
                List<List<double>> indexes;

                indexes = oComb.GenerateCombinations(tmpList, 
                                        oConfiguration.NumberOfTasksPerSet);

                foreach(List<double> lst in indexes)
                {
                    ps = new CPeriodSet();
                    foreach (double p in lst)
                        ps.Add(p);
                    _allPeriods.Add(ps);
                }

            }

            return _allPeriods;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="oConfiguration"></param>
        private void GeneratePrimes(CConfiguration oConfiguration)
        {
            double delta = oConfiguration.MaxPeriod - oConfiguration.MinPeriod + 1;
            IPrimeNumberGenerator oPrime = new CSieveOfEratosthenes();
            _primes = oPrime.GeneratePrimeNumbers(0,delta);//, oConfiguration.MaxPeriod);
        }


        /// <summary>
        /// Generates all the periods of a sepcific group
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="prime"></param>
        /// <returns></returns>
        private List<double> GeneratePeriods(double min, double max, double prime)
        {
            double multiple;
            List<double> tmp = new List<double>();
            //Start from min
            //Find first divisor for prime
            //Then find all divisors till max and add them to List
            multiple = min;
            while (multiple % prime != 0) { multiple++; }


            while (multiple <= max)
            {
                tmp.Add(multiple);
                multiple += prime;
            }

            return tmp;
        }


        /// <summary>
        /// Generate all possible unique sets of iSizeOfSubet from elements inside mainSet
        /// </summary>
        /// <param name="listOfElement"></param>
        /// <param name="iSizeOfCombinations"></param>
        //private void GenerateCombinations(List<double> mainSet, int iSizeOfSubSet)
        //{
        //    double multiple;
        //    List<double> tmp = new List<double>();
        //    //Start from min
        //    //Find first divisor for prime
        //    //Then find all divisors till max and add them to List
        //    multiple = min;
        //    while (multiple % prime != 0) { multiple++; }


        //    while (multiple <= max)
        //    {
        //        tmp.Add(multiple);
        //        multiple += prime;
        //    }

        //    return tmp;

        //}


        /// <summary>
        /// Get Maximum Possible Unique Period Sets
        /// </summary>
        /// <param name="oConfiguration"></param>
        /// <returns></returns>

        

        public long GetMaximumUniquePeriodSets(CConfiguration oConfiguration)
        {
            int idxMaxGroups;
            long noOfFactors;
            long lMaxPeriodSets = 0;
            CCombinationGenerator comb = new CCombinationGenerator();
            
            int idx;

            if (_primes == null) GeneratePrimes(oConfiguration);

            idxMaxGroups = GetMaximumGroups(oConfiguration.MinPeriod, oConfiguration.MaxPeriod,
                oConfiguration.NumberOfTasksPerSet);
            idxMaxGroups--; //idxMaxGroups = 2

            
            
            //for(idx = 0;idx <= idxMaxGroups;idx++)
            for (idx = idxMaxGroups; idx > 0; idx--)
            {
                noOfFactors = GetNumberOfFactors(_primes[idx-1], oConfiguration.MinPeriod, oConfiguration.MaxPeriod);
                lMaxPeriodSets = lMaxPeriodSets + (long) comb.GetNumberOfCombinations(noOfFactors,
                                    oConfiguration.NumberOfTasksPerSet);
            }

            return lMaxPeriodSets;
        }


        /// <summary>
        /// Calculates the number of factors for a given prime
        /// </summary>
        /// <param name="prime"></param>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        private long GetNumberOfFactors(double prime, double min, double max)
        {
            double maxMultiple,minMultiple;
            long tmp;

            //Start from min
            //Find first divisor for prime
            //Then find all divisors till max and add them to List
            minMultiple = min;
            while (minMultiple % prime != 0) { minMultiple++; }
            maxMultiple = max;
            while (maxMultiple % prime != 0) { maxMultiple--; }

            tmp = ((long)(maxMultiple - minMultiple) /(long)prime) + 1;
            
            return tmp;
        }

        

       

        /// <summary>
        /// In the range [5, 10] it should return  
        /// </summary>
        /// <param name="max"></param>
        /// <param name="min"></param>
        /// <returns></returns>
        private double GetMaximumFactor(double min, double max, int tasksPerSet)
        {
            double maxFactor;
            
            // (taskSetSize-1) * X + min <= max
            // (taskSetSize-1) * X = max - min
            // X = Ceiling(max - min / (taskSetSize-1))
            // e.g. max = 10, min = 5, taskSetSize = 2
            // maxFactor = 5;
            // If taskSetSize = 3;max Factor = 2;
            maxFactor = Math.Ceiling((max - min) / ((double)tasksPerSet - 1.0));

            return maxFactor;
        }

        /// <summary>
        /// Gets the value of the maximum prime which is less than maxPrime
        /// </summary>
        /// <param name="maxPrime"></param>
        /// <returns></returns>
        private int GetIndexOfMaxPrime(double maxPrime)
        {
            // e.g. max = 10, min = 5, taskSetSize = 2
            // maxPrime = 5;
            // Primes: 2,3,5,7,9
            // return idx = 2
            int idx = 0;
            for (idx = 0; idx < _primes.Count; idx++)
            //for (idx = _primes.Count - 1; idx >= 0; idx--)
                if (_primes[idx] > maxPrime)
                {
                    if (idx - 1 < 0) return -1;
                    //return (idx - 1);
                    return (idx + 1);
                }

            return idx +1;
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <param name="tasksPerSet"></param>
        /// <returns></returns>
        private int GetMaximumGroups(double min,double max, int tasksPerSet) 
        {
            int idxMaxPrime;

            double maxFactor = GetMaximumFactor(min, max, tasksPerSet);

            // idx = 2 ; maxGRoups = 2 + 1 (Because of 0 Index)

            idxMaxPrime = GetIndexOfMaxPrime(maxFactor);
            if (idxMaxPrime < 0) return 0;
            else return (idxMaxPrime + 1);
        }

        public string Description
        {
            get { return "All Bounded Periods using Combinatorial Arithmetic"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }
        

        
    }
}
