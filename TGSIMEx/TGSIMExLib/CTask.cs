using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public class CTask
    {
        private double _Period;
        private double _ExecutionTime;
        private double _Offset;
        private double _RelativeDeadline;
        private string _sID;

        public double dBlocking;
        public int iPriority;
        public double dExecutedSteps;
        public double dScratch;
        public int iScratch;
        
        

        public CTask(string sID, 
                     int iPriority,
                     double dPeriod, 
                     double dExecutionTime, 
                     double dOffset, 
                     double dRelativeDeadline)
        {
            _sID = sID;
            _Period = dPeriod;
            _ExecutionTime = dExecutionTime;
            _Offset = dOffset;
            _RelativeDeadline = dRelativeDeadline;
            this.iPriority = iPriority;
        }

        public string ID
        {
            get { return _sID; }
        }

        public double ExecutionTime
        {
            get { return _ExecutionTime; }
        }

        public double Period
        {
            get { return _Period; }
        }

        public double Offset
        {
            get { return _Offset; }
        }

        public double Utilization
        {
            get { return (ExecutionTime / Period); }
        }

        public double RelativeDeadline
        {
            get { return _RelativeDeadline; }
        }


    }
}
