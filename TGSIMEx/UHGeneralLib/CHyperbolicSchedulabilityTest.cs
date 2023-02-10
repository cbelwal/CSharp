using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CHyperbolicSchedulabilityTest:ISchedulabilityTest
    {
        private string _message;

        public string Description
        {
            get { return "Bini et al's Hyperbolic Bound Test"; }
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
            try
            {
                double dUtil = 1;
                CTask[] taskArray = taskSet.GetAllTasksInPriorityOrder();

                foreach (CTask t in taskArray)
                {
                    dUtil = dUtil * (t.Utilization + 1.0);
                }

                if (dUtil <= 2) return true;
                else return false;
            }
            catch(Exception e)
            {
                _message = e.Message;
                return false;

            }
        }
    }
}
