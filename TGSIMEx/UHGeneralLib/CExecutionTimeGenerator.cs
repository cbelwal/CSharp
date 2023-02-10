using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CExecutionTimeGenerator:IExecutionTimeGenerator
    {
        private Random random;
        private string _message;

        public CExecutionTimeGenerator()
        {
            random = new Random();
        }

        public string Description
        {
            get { return "Randomly generate Execution Times within bounded range"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }

        public string Message
        {
            get { return _message; }
        }

        /// <summary>
        /// Return Execution Times
        /// </summary>
        /// <param name="configuration"></param>
        /// <returns></returns>
        public List<double> GetExecutionTimes(CConfiguration configuration)
        {
            long tmpValue;
            long maxPossibleValues = (long)configuration.MaxExecutionTime - (long)configuration.MinExecutionTime + 1;
            long noValuesToGenerate;


            
            List<double> lstExecutionTimes = new List<double>();
            try
            {
                if (maxPossibleValues < 2*configuration.NumberOfTasksPerSet)
                    noValuesToGenerate = maxPossibleValues;
                else
                {
                    noValuesToGenerate = maxPossibleValues / 2;
                }


                while (lstExecutionTimes.Count < noValuesToGenerate)
                {
                    tmpValue = GetRandomValue((long)configuration.MinExecutionTime, (long)configuration.MaxExecutionTime);

                    if (!(configuration.bUniqueExecutionTimes && lstExecutionTimes.Contains(tmpValue)))
                    {
                        lstExecutionTimes.Add(tmpValue);
                    }
                }
            }
            catch (Exception e)
            {
                _message = e.Message;
            }


            return lstExecutionTimes;
        }

        /// <summary>
        /// Returns a random period
        /// </summary>
        /// <param name="minPeriod"></param>
        /// <param name="maxPeriod"></param>
        /// <returns></returns>
        private long GetRandomValue(long min, long max)
        {
            long Value;
            Value = (long)random.Next((int)min, (int)max+1); //Since upper bound is exclusive
            return Value;
        }

    }
}
