using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    /// <summary>
    /// 
    /// </summary>
    public class CConfiguration
    {
        private double _MaxPeriod;
        private double _MinPeriod;

        private double _MaxOffset;
        private double _MinOffset;

        private double _MinExecutionTime;
        private double _MaxExecutionTime;

        private int _NumberOfTasksPerSet;
        private int _NumberOfTaskSets;
        private static string _message;

        public bool bUniquePeriods;
        public bool bUniqueExecutionTimes;
        public bool bRunSimulation;
        


        private CConfiguration (int iNumberOfTaskSets,
                              int iNumberOfTasksPerSet,
                              double dMinPeriod,
                              double dMaxPeriod,
                              double dMinExecutionTime,
                              double dMaxExecutionTime,
                              double dMinOffset,
                              double dMaxOffset)
        {
            _NumberOfTasksPerSet = iNumberOfTasksPerSet;
            _NumberOfTaskSets = iNumberOfTaskSets;

            _MaxPeriod = dMaxPeriod;
            _MinPeriod = dMinPeriod;

            _MinExecutionTime = dMinExecutionTime;
            _MaxExecutionTime = dMaxExecutionTime;

            _MaxOffset = dMaxOffset;
            _MinOffset = dMinOffset;
        }

        /// <summary>
        /// Creates the Configuration Object
        /// </summary>
        /// <param name="iNumberOfTaskSets"></param>
        /// <param name="iNumberOfTasksPerSet"></param>
        /// <param name="dMinPeriod"></param>
        /// <param name="dMaxPeriod"></param>
        /// <param name="dMinExecutionTime"></param>
        /// <param name="dMaxExecutionTime"></param>
        /// <param name="dMinOffset"></param>
        /// <param name="dMaxOffset"></param>
        /// <returns></returns>
        public static CConfiguration GetObject(int iNumberOfTaskSets,
                              int iNumberOfTasksPerSet,
                              double dMinPeriod,                  
                              double dMaxPeriod,
                              double dMinExecutionTime,
                              double dMaxExecutionTime,
                              double dMinOffset,
                              double dMaxOffset)
        {
            _message = "";
            if (dMaxPeriod - dMinPeriod + 1 < iNumberOfTasksPerSet)
            {
                _message = "The range between minimum and maximum Periods is too low";
                return null;
            }

            if (dMaxExecutionTime - dMinExecutionTime + 1 < 2)
            {
                _message = "The range between minimum and maximum execution time is too low";
                return null;
            }
            
            CConfiguration config = new CConfiguration(iNumberOfTaskSets,
                                                        iNumberOfTasksPerSet,
                                                        dMinPeriod,                  
                                                        dMaxPeriod,
                                                        dMinExecutionTime,
                                                        dMaxExecutionTime,
                                                        dMinOffset,
                                                        dMaxOffset);
            return config;
        }

        public static string Message
        {
            get { return _message; }
        }

        public double MaxExecutionTime
        {
            get { return _MaxExecutionTime; }
        }

        public double MinExecutionTime
        {
            get { return _MinExecutionTime; }
        }


        public double MaxPeriod
        {
            get { return _MaxPeriod; }
        }

        public double MinPeriod
        {
            get { return _MinPeriod; }
        }

        public double MaxOffset
        {
            get { return _MaxOffset; }
        }

        public double MinOffset
        {
            get { return _MinOffset; }
        }

        public int NumberOfTasksPerSet
        {
            get { return _NumberOfTasksPerSet; }
        }

        public int NumberOfTaskSets
        {
            get { return _NumberOfTaskSets; }
        }

        public long NumberOfDiscretePeriods
        {
            get { return (long)(MaxPeriod - MinPeriod + 1); }
        }
    }
}
