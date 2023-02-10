using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using TGSIMExLib;
using UHGeneralLib;

namespace UHGeneralLibTest
{
    public partial class fMain : Form
    {
        private List<ISchedulabilityTest> schedTests;
        private List<ISatisfiabilityCondition> satConditions;
        private CConfiguration config;
        private List<CTaskSet> allTaskSets;
        

        public fMain()
        {
            InitializeComponent();
        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            Thread newThread = new Thread(CreateTaskSets);
            newThread.Start();
            
            //Start in New Thread

        }


        private void CreateTaskSets()
        {
            bool bStatus;
            allTaskSets = new List<CTaskSet>();

            //Create Configuration
            config = CConfiguration.GetObject(20, 3, 3, 40, 1, 7, 0, 0);
            config.bUniqueExecutionTimes = true;

            //Add Sched Tests
            schedTests = new List<ISchedulabilityTest>();
            ISchedulabilityTest sch1 = new CHyperbolicSchedulabilityTest();
            ISchedulabilityTest sch2 = new CLiuLaylandSchedulabilityTest();
            ISchedulabilityTest sch3 = new CUtilizationSchedulabilityTest();

            schedTests.Add(sch1);
            
            schedTests.Add(sch2);
            schedTests.Add(sch3);

            //Add Satisfiability Conditions
            satConditions = new List<ISatisfiabilityCondition>();
            ISatisfiabilityCondition sat1 = new CPFRPSatisfiabilityCondition();
            satConditions.Add(sat1);
            
            //Generate Period Sets and Execution Times
            List<CPeriodSet> _periodSets = GetPeriods(config);
            List<double> _executionTimes = GetExecutionTime(config);
   
            //Generate Release Offsets
            List<double> _releaseOffsets = GetReleaseOffsets(config);


            //Create Callbacks
            CCallBack cb = new CCallBack();
            cb.funcReceivedTaskSet += CheckSchedulability;


            //Call Task Set Generator
            CTaskSetGenerator tsg = new CTaskSetGenerator();
            tsg.GenerateTaskSets(_periodSets, _executionTimes, _releaseOffsets, config, cb);

            //Write Task Set Files
            CSimpleFileFormatter ff = new CSimpleFileFormatter();
            bStatus = ff.WriteFiles(allTaskSets,Application.StartupPath + "\\Files",config);


            if (bStatus) MessageBox.Show("All Files Successively Written");

        }

        /// <summary>
        /// This Function is Called through deletegate in CCallBack;
        /// </summary>
        /// <param name="taskSet"></param>
        /// <returns></returns>
        private bool CheckSchedulability(CTaskSet taskSet)
        {
            if (!CheckSchedulabililityTests(schedTests, taskSet, config)) return false;
            if (!CheckSatisfiabilityTests(satConditions, taskSet, config)) return false;

            //Add Task Set to Master List

            allTaskSets.Add(taskSet);
            return true;

        }

        private bool CheckSchedulabililityTests(List<ISchedulabilityTest> schedTests,
                                                CTaskSet taskSet, CConfiguration config)
        {
            foreach (ISchedulabilityTest test in schedTests)
            {
                if (!test.IsSatisfied(taskSet, config)) return false;
            }
            return true;
        }

        private bool CheckSatisfiabilityTests(List<ISatisfiabilityCondition> satConds,
                                                CTaskSet taskSet, CConfiguration config)
        {
            foreach (ISatisfiabilityCondition cond in satConds)
            {
                if (!cond.IsSatisfied(taskSet, config)) return false;
            }
            return true;
        }
        


        private List<double> GetExecutionTime(CConfiguration config)
        {
            CExecutionTimeGenerator etg = new CExecutionTimeGenerator();
            return etg.GetExecutionTimes(config);
        }


        private List<double> GetReleaseOffsets(CConfiguration config)
        {
            CReleaseOffsetGenerator rog = new CReleaseOffsetGenerator();
            return rog.GetReleaseOffsets(config);
        }


        //Generate Periods
        private List<CPeriodSet> GetPeriods(CConfiguration config)
        {

            List<CPeriodSet> allPeriods = new List<CPeriodSet>();

            CPeriodSet ps = new CPeriodSet();
            ps.Add(10);
            ps.Add(12);
            ps.Add(15);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(18);
            ps.Add(20);
            ps.Add(21);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(24);
            ps.Add(27);
            ps.Add(30);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(31);
            ps.Add(32);
            ps.Add(33);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(36);
            ps.Add(29);
            ps.Add(42);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(45);
            ps.Add(48);
            ps.Add(51);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(54);
            ps.Add(57);
            ps.Add(60);
            allPeriods.Add(ps);

            ps = new CPeriodSet();
            ps.Add(63);
            ps.Add(66);
            ps.Add(69);
            allPeriods.Add(ps);


            return allPeriods;

        }
    }
}
