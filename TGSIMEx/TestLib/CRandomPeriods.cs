using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using TGSIMExLib;


namespace UHPeriodLib
{
    public class CRandomPeriods : IPeriodGenerator
    {
        Random random;
        private string _Message;

        public string Message
        {
            get { return _Message; }
        }

        public CRandomPeriods()
        {
            random = new Random();
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="oConfiguration"></param>
        /// <returns></returns>
        public List<CPeriodSet> GetPeriods(CConfiguration oConfiguration, 
                            IPrimeNumberGenerator primeGenerator)
        {
            int noOfPeriods = oConfiguration.NumberOfTaskSets;
            Hashtable hshMain = new Hashtable();
            long tmp;
            List<CPeriodSet> allPeriodSets = new List<CPeriodSet>();
            CPeriodSet ps;

            _Message = "";

            while (allPeriodSets.Count < oConfiguration.NumberOfTaskSets)
            {
                ps = new CPeriodSet();

                while (ps.Count < oConfiguration.NumberOfTasksPerSet)
                {
                    tmp = GetRandomPeriod((long)oConfiguration.MinPeriod, (long)oConfiguration.MaxPeriod);
                    while (ps.isPresent(tmp))
                        tmp = GetRandomPeriod((long)oConfiguration.MinPeriod, (long)oConfiguration.MaxPeriod);

                    ps.Add(tmp);
                }

                allPeriodSets.Add(ps);
            }

            return allPeriodSets;
        }


        /// <summary>
        /// Returns a random period
        /// </summary>
        /// <param name="minPeriod"></param>
        /// <param name="maxPeriod"></param>
        /// <returns></returns>
        private long GetRandomPeriod(long minPeriod,long maxPeriod)
        {
            long period;
            
            period = (long)random.Next((int)minPeriod, (int)maxPeriod);

            return period;
        }

        public long GetMaximumUniquePeriodSets(CConfiguration oConfiguration)
        { return oConfiguration.NumberOfTaskSets; }

        public string Description
        {
            get { return "Random Period Generator"; }

        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }
    }
}
