using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace TaskGen
{
    class CInternalConfiguration
    {
        //public List<CListBoxWrapper> allSchedulabilityTests;
        //public List<CListBoxWrapper> allSatisfiabilityConditions;
        //public List<CListBoxWrapper> allFileFormattersTests;

        public List<CListBoxWrapper> selectedSchedulabilityTestWrappers;
        public List<CListBoxWrapper> selectedSatisfiabilityConditionWrappers;
       
        public CListBoxWrapper selectedFileFormatter;
        public CListBoxWrapper selectedExecutionTraceWriter;

        public CListBoxWrapper selectedExecutionModel;
        public CListBoxWrapper selectedPeriodGenerator;
        public CListBoxWrapper selectedExecutionTimeGenerator;
        public CListBoxWrapper selectedReleaseOffsetGenerator;
        public CListBoxWrapper selectedTaskSetGenerator;

        public CInternalConfiguration()
        {
            //allSchedulabilityTests = null;
            //allSatisfiabilityConditions = null;
            //allFileFormattersTests = null;

            selectedSchedulabilityTestWrappers = new List<CListBoxWrapper>();
            selectedSatisfiabilityConditionWrappers = new List<CListBoxWrapper>();
            selectedFileFormatter = null;
            selectedExecutionTraceWriter = null;

            selectedExecutionModel = null;
            selectedPeriodGenerator = null;
            selectedExecutionTimeGenerator = null;
            selectedReleaseOffsetGenerator = null;
            selectedTaskSetGenerator = null;
            

        }


        /// <summary>
        /// If return value is "" all Input are ready
        /// </summary>
        /// <returns></returns>
        public string verifyInputReadyForTaskGeneration()
        {
            string sMessage="";

            if (selectedPeriodGenerator == null) sMessage = "No Period Generator Selected";
            if (selectedExecutionTimeGenerator == null) sMessage = "No Execution Time Generator Selected";
            if (selectedReleaseOffsetGenerator == null) sMessage = "No Release Offset Generator Selected";
            if (selectedTaskSetGenerator == null) sMessage = "No Task Set Generator Selected";
            if (selectedFileFormatter == null) sMessage = "No File Formatter Selected";

            //Tasks can be generated with Scheduling or Satisfiability conditions

            return sMessage;
        }

        public string verifyInputReadyForSimulation()
        {
            string sMessage = "";

            if (selectedExecutionModel.objMain == null) sMessage = "No Execution Model Selected for Simulation to Run";

            return sMessage;
        }

        public List<ISchedulabilityTest> SchedulabilityTests
        {
            get
            {
                List<ISchedulabilityTest> schedTests = new List<ISchedulabilityTest>();
                foreach (CListBoxWrapper oWrap in selectedSchedulabilityTestWrappers)
                {
                    schedTests.Add((ISchedulabilityTest) oWrap.objMain);
                }
                return schedTests;
            }
        }

        
        public List<ISatisfiabilityCondition> SatisfiabilityConditions
        {
            get
            {
                List<ISatisfiabilityCondition> satConditions = new List<ISatisfiabilityCondition>();
                foreach (CListBoxWrapper oWrap in selectedSatisfiabilityConditionWrappers)
                {
                    satConditions.Add((ISatisfiabilityCondition) oWrap.objMain);
                }
                return satConditions;
            }
        }


    }
}
