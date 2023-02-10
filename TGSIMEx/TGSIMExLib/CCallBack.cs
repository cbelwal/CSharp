using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public class CCallBack
    {

        public delegate bool ReceivedTaskSet(CTaskSet taskSet);
        public ReceivedTaskSet funcReceivedTaskSet;

        public bool AddGeneratedTaskSet(CTaskSet taskSet)
        {
            bool bStatus;
            //Call the delegate in main routine
            bStatus = funcReceivedTaskSet(taskSet);

            return bStatus;
        }
    }
}
