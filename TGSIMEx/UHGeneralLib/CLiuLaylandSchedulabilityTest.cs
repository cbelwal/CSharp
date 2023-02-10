using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CLiuLaylandSchedulabilityTest:ISchedulabilityTest
    {
        private string _message;
        /// <summary>
        /// Returns true / false depending on the Schedulability Condition
        /// </summary>
        /// <param name="taskSet"></param>
        /// <param name="configuration"></param>
        /// <returns></returns>
        public bool IsSatisfied(CTaskSet taskSet, CConfiguration configuration)
        {
            double LLB;

            try
            {
                CTask[] allTasks = taskSet.GetAllTasksInPriorityOrder();
                double U = taskSet.Utilization;


                LLB = taskSet.Count * (Math.Pow(2.0, (1/(double)taskSet.Count)) - 1);

                if (U < LLB) return true;
                else return false;
            }
            catch (Exception e)
            {
                _message = e.Message;
                return false;
            }

        }

        
        public string Description
        {
            get { return "Liu and Layland's Utilization based Test"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }

        public string Message
        {
            get { return _message; }
        }
    }
}
