using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public class CPeriodSet:IComparable<CPeriodSet>
    {
        private List<double> periods;


        public int CompareTo(CPeriodSet ps)
        {
            return (int) (GetLCM() - ps.GetLCM());

        }

        public CPeriodSet()
        {
            periods = new List<double>();
        }

        
        public bool Add(double dPeriod)
        {
            try
            {
                periods.Add(dPeriod);
                periods.Sort();
                return true;
            }
            catch
            {
                return false;
            }
        }

        public bool Delete(double dPeriod)
        {
            periods.Sort();
            return true;
        }

        public bool isPresent(double dPeriod)
        {
            foreach (double p in periods)
                if (p == dPeriod) return true;

            return false;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="index"></param>
        /// <returns></returns>
        public double Get(int index)
        {
            try
            {
                return periods[index];
            }
            catch
            {
                return 0; // periods cant be 0
            }
        }

        /// <summary>
        /// 
        /// </summary>
        public int Count
        {
            get{return periods.Count;}
        }

        /// <summary>
        /// Returns the Hash Code of all Periods
        /// </summary>
        /// <returns></returns>
        public string GetHash()
        {
            string sPeriods="";
            //Sort the Items

            foreach (double p in periods)
                sPeriods = sPeriods + p.ToString();
            
            return sPeriods;   
        }

        /// <summary>
        /// 
        /// </summary>
        /// <returns></returns>
        public string GetPeriodString()
        {
            string sTmp="";
            foreach (double d in periods)
            {
                sTmp = sTmp + d.ToString() + ", ";
            }

            //Remove last ,
            sTmp = sTmp.Remove(sTmp.Length - 2, 2);
            return sTmp;
        }

        /// <summary>
        /// Returns the LCM of all numbers
        /// </summary>
        /// <returns></returns>
        public double GetLCM()
        {
            return CMath.LCM(periods);            
        }

    }
}
