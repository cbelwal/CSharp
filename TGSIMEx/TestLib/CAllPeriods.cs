using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHPeriodLib
{
    public class CAllPeriods : IPeriodGenerator
    {
        public CAllPeriods() { }
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
        public List<CPeriodSet> GetPeriods(CConfiguration oConfiguration, IPrimeNumberGenerator primeGenerator)
        {
            List<List<double>> allCombinations;
            List<CPeriodSet> allPeriodSets = new List<CPeriodSet>();
            List<double> mainSet = generateMainSet(oConfiguration.MinPeriod, oConfiguration.MaxPeriod);

            CCombinationGenerator comb = new CCombinationGenerator();

            _Message = "";

            allCombinations = comb.GenerateCombinations(mainSet, oConfiguration.NumberOfTasksPerSet); 
  


            //Convert allCombinations to allPeriodSets
            CPeriodSet ps;
            foreach (List<double> c in allCombinations)
            {
                ps = new CPeriodSet();

                foreach (double d in c)
                    ps.Add(d);

                allPeriodSets.Add(ps);
            }


            return allPeriodSets;
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="oConfiguration"></param>
        /// <returns></returns>
        public long GetMaximumUniquePeriodSets(CConfiguration oConfiguration)
        {

            long lMaxPeriodSets;
            
            CCombinationGenerator comb = new CCombinationGenerator();

            lMaxPeriodSets = (long)comb.GetNumberOfCombinations(oConfiguration.NumberOfDiscretePeriods,
                                    oConfiguration.NumberOfTasksPerSet);

            return lMaxPeriodSets;

        }

        /// <summary>
        /// Generate all values possible between min and max
        /// </summary>
        /// <param name="min"></param>
        /// <param name="max"></param>
        /// <returns></returns>
        private  List<double> generateMainSet(double min, double max)
        {
            double ii;
            List<double> set = new List<double>();

            for (ii = min; ii <= max; ii++)
                set.Add(ii);

            return set;
        }

        public string Description
        {
            get { return "Bounded Periods with Low Hyperperiods using Prime Numbers and Combinatorial Arithmetic"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }
    }
}
