using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CReleaseOffsetGenerator : IReleaseOffsetGenerator
    {
        private string _message;
        
        public string Description
        {
            get { return "All Release Offsets are 0 (for Offset Free Systems)"; }
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
        /// Return Execution Times all equal to 0
        /// </summary>
        /// <param name="configuration"></param>
        /// <returns></returns>
        public List<double> GetReleaseOffsets(CConfiguration configuration)
        {
            int ii;
            List<double> ro = new List<double>();

            try
            {
                

                for (ii = 0; ii < configuration.NumberOfTasksPerSet; ii++)
                    ro.Add(0);

                
            }
            catch (Exception e)
            {
                _message = e.Message;
            }
            return ro;
        }


    }
}
