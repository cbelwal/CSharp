using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CPFRPSatisfiabilityCondition:ISatisfiabilityCondition
    {
        private string _message;

        public string Description
        {
            get { return "P-FRP Execution Model Necessary Condition"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }

        public string Message
        {
            get { return _message; }
        }

        public bool IsSatisfied(CTaskSet taskSet, CConfiguration configuration)
        {
            long slackTime;

            try
            {
                CTask[] tasks = taskSet.GetAllTasksInPriorityOrder();

                //Now check different in periods
                foreach (CTask t in tasks)
                {
                    slackTime = (long)t.Period - (long)t.ExecutionTime;


                    //Now make sure the processing time for every event in the set has a lesser slack time
                    foreach (CTask t1 in tasks)
                    {
                        if (t1.ID != t.ID)
                        {
                            if (t1.ExecutionTime > slackTime) return false;
                        }
                    }
                }

                return true;
            }
            catch(Exception e)
            {
                _message = e.Message;
                return false;

            }
        }

    }
}
