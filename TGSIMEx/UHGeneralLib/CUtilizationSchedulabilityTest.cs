using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CUtilizationSchedulabilityTest :ISchedulabilityTest
    {
        private string _message;

        public string Description
        {
            get { return "Utilization <= 1 Test"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }

        public bool IsSatisfied(CTaskSet taskSet, CConfiguration configuration)
        {
            try
            {
                double U = taskSet.Utilization;
                if (U <= 1) return true;
                else return false;
            }
            catch (Exception e)
            {
                _message = e.Message;
                return false;
            }
        }

        public string Message
        {
            get { return _message; }
        }

    }
}
