using System;
using System.Collections.Generic;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using System.IO;
using TGSIMExLib;


namespace TaskGen
{
    public partial class fMain : Form
    {
        //Declare Private Variables
        private List<CTaskSet> _allTaskSets;
        private Thread activeThread;
        private CInternalConfiguration _internalConfig;
        private int _taskCount;
        private List<ISchedulabilityTest> _schedTests;
        private List<ISatisfiabilityCondition> _satConditions;
        private CConfiguration _config;
        private delegate void UpdateStatusDelegate(string sMessage);
        private delegate void AddRunTimeMessageDelegate(string sMessage);
        private bool _forceStop;

        public fMain()
        {
            InitializeComponent();
            _internalConfig = new CInternalConfiguration();
        }


        private void btnBrowse_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.SelectedPath = Application.StartupPath;
            folderBrowserDialog1.ShowDialog();
            textFolderName.Text = folderBrowserDialog1.SelectedPath;
        }



        private void UpdateStatus(string sMessage)
        {
            

            if (statusStrip1.InvokeRequired)
            {
                statusStrip1.Invoke(new UpdateStatusDelegate(this.UpdateStatus), sMessage);
            }
            else
                toolStripStatusLeft.Text = sMessage.Trim();

            Application.DoEvents();
        }

        /// <summary>
        /// Put all Plug Ins in specified List boxes
        /// </summary>
        /// <param name="typeObj"></param>
        /// <param name="listBox"></param>
        private void ListPlugIns(Type typeObj, CheckedListBox listBox)
        {
            ArrayList allObjs;
            CObjectFactory factory = new CObjectFactory();
            //IRootInterface oTmp;
            //String sTmp;
            CListBoxWrapper oWrap;

            allObjs = factory.GetObjects(typeObj, Application.StartupPath);
            AddRunTimeMessage("Add PlugIns", factory.errorMessages);

            if (allObjs == null) return;

            foreach (Object o in allObjs)
            {
                oWrap = new CListBoxWrapper();
                oWrap.objMain= (IRootInterface)o;
                oWrap.index = listBox.Items.Count;
                listBox.Items.Add(oWrap);
            }

        }

     
        private void fMain_Load(object sender, EventArgs e)
        {
           Initialize();
           //simulationToolStripMenuItem_Click(sender,e);
        }


        /// <summary>
        /// Initialize
        /// </summary>
        private void Initialize()
        {
            UpdateStatus("Initializing");

            ListPlugIns(typeof(ISchedulabilityTest), checkedListBoxSchedulabilityTests);
            ListPlugIns(typeof(ISatisfiabilityCondition), checkedListBoxSatisfiabilityConditions);
            ListPlugIns(typeof(IFileFormatter), checkedListBoxFileFormatter);

            textFolderName.Text = Application.StartupPath + "\\Files";

            UpdateStatus("Ready");
        }



        private void btnStart_Click(object sender, EventArgs e)
        {
            //Internal Configuration  for Testing -> REMOVE in Production
            SetTestConfiguration();

            ResetRunTimeMessages();
            _forceStop = false;
            CThreadObject to = new CThreadObject();
            to.config = GetConfigurationObject();

            if (to.config == null) return; //STOP

            to.folderPathforFiles = textFolderName.Text.Trim();

            if (to.folderPathforFiles.Trim() == "")
            {
                MessageBox.Show("Please enter valid folder name");
                return;
            }

            //Initialize Internal Config
            if (!InitializeInternalConfig(to))
            {
                UpdateStatus("Ready");
                return; //Return if not set
            }

            //Activate Thread
            SetActiveUI(false);
            Thread newThread = new Thread(StartTaskGeneration);
            newThread.Start(to);
            activeThread = newThread;

            while (activeThread.IsAlive) { Application.DoEvents(); }

            AddRunTimeMessage("Task Generation Complete");
            UpdateStatus("Task Generation Complete");

            if (!_forceStop)
            {
                //Auto Start Simulation if Required
                if (checkRunSimulation.Checked)
                    AutoStartSimulation();

            }
            SetActiveUI(true);
            UpdateStatus("Ready");

        }

        private bool InitializeInternalConfig(CThreadObject to)
        {
            string msg;
            InitTestsAndConditions(to);

            //Add File Formatter
            foreach (CListBoxWrapper oWrap in checkedListBoxFileFormatter.CheckedItems)
            {
                _internalConfig.selectedFileFormatter = oWrap;
                break; //Add Only One
            }


            //Do Input Validation
            msg = _internalConfig.verifyInputReadyForTaskGeneration();

            if (msg.Trim() != "")
            {   //Some Inputs are Not Complete
                MessageBox.Show(msg);
                return false;
            }
            return true;
        }


        /// <summary>
        /// GetConfigurationObject
        /// </summary>
        /// <returns></returns>
        private CConfiguration GetConfigurationObject()
        {
            CConfiguration config =  CConfiguration.GetObject( Convert.ToInt32( textNoOfTaskSets.Text), 
                                                        Convert.ToInt32(textNoOfTasks.Text), 
                                                        Convert.ToDouble( textMinimumPeriod.Text),
                                                        Convert.ToDouble(textMaximumPeriod.Text),
                                                        Convert.ToDouble(textMinimumExecutionTime.Text),
                                                        Convert.ToDouble(textMaximumExecutionTime.Text),
                                                        Convert.ToDouble(textMinimumOffset.Text),
                                                        Convert.ToDouble(textMaximumOffset.Text));


            if (CConfiguration.Message.Trim() != "")
            {
                MessageBox.Show(CConfiguration.Message);
                config = null;
                return config;
            }
            
            config.bRunSimulation = checkRunSimulation.Checked;
            config.bUniqueExecutionTimes = checkUniqueExecutionTime.Checked;
            config.bUniquePeriods = checkUniquePeriod.Checked;

            return config;
        }

        /// <summary>
        /// Sets up configuration for Testing
        /// </summary>
        private void SetTestConfiguration()
        {
            //Generate Wrappers
            _internalConfig.selectedExecutionModel = new CListBoxWrapper();
            _internalConfig.selectedExecutionTimeGenerator = new CListBoxWrapper();
            _internalConfig.selectedFileFormatter = new CListBoxWrapper();
            _internalConfig.selectedPeriodGenerator = new CListBoxWrapper();
            _internalConfig.selectedReleaseOffsetGenerator = new CListBoxWrapper();
            _internalConfig.selectedSatisfiabilityConditionWrappers.Add(new CListBoxWrapper());
            _internalConfig.selectedSchedulabilityTestWrappers.Add(new CListBoxWrapper());
            //_internalConfig.selectedSchedulabilityTestWrappers.Add(new CListBoxWrapper());
            //_internalConfig.selectedSchedulabilityTestWrappers.Add(new CListBoxWrapper());
            _internalConfig.selectedTaskSetGenerator = new CListBoxWrapper();
            _internalConfig.selectedExecutionTraceWriter = new CListBoxWrapper();

            //Generate Actual Model
            _internalConfig.selectedExecutionModel.objMain =   new UHGeneralLib.CPFRPExecutionModel();
            //_internalConfig.selectedExecutionModel.objMain = new UHGeneralLib.CPreemptiveExecutionModel();
            _internalConfig.selectedExecutionTimeGenerator.objMain = new UHGeneralLib.CExecutionTimeGenerator();
            //_internalConfig.selectedFileFormatter.objMain = new UHGeneralLib.CSimpleFileFormatter();
            _internalConfig.selectedFileFormatter.objMain = new UHGeneralLib.CSimpleFileFormatterReverseOrder();
            _internalConfig.selectedPeriodGenerator.objMain = new UHPeriodLib.CBoundedPeriods();
            _internalConfig.selectedReleaseOffsetGenerator.objMain = new UHGeneralLib.CReleaseOffsetGenerator();
            _internalConfig.selectedSatisfiabilityConditionWrappers[0].objMain = new UHGeneralLib.CPFRPSatisfiabilityCondition();
            //_internalConfig.selectedSchedulabilityTestWrappers[0].objMain = new UHGeneralLib.CLiuLaylandSchedulabilityTest();
            //_internalConfig.selectedSchedulabilityTestWrappers[1].objMain =  new UHGeneralLib.CLiuLaylandSchedulabilityTest();
            _internalConfig.selectedSchedulabilityTestWrappers[0].objMain = new UHGeneralLib.CUtilizationSchedulabilityTest();
            _internalConfig.selectedTaskSetGenerator.objMain = new UHGeneralLib.CTaskSetGenerator();
            _internalConfig.selectedExecutionTraceWriter.objMain = new UHGeneralLib.CExecutionTraceWriter();           
        }

        private void ResetRunTimeMessages()
        {
            textBoxMessages.Text = "";

        }

        //This Function Runs in a new Thread
        private void StartTaskGeneration(Object threadObject)
        {
            bool bStatus;
            CThreadObject to = (CThreadObject)threadObject;
            CConfiguration config = to.config;
            
            
            List<CPeriodSet> periodSets;
            List<double> executionTimes;
            List<double> releaseOffsets;
            
            //Set Global Vars
            _config = config;
            _taskCount = 1;
            _allTaskSets = new List<CTaskSet>();

            

           
           
            //Generate Period Sets
            UpdateStatus("Calling Period Generator ...");

            IPeriodGenerator pg = (IPeriodGenerator)_internalConfig.selectedPeriodGenerator.objMain;
            
            //try
            //{
                periodSets = pg.GetPeriods(config, null);
            //}
            //catch (Exception e)
            //{
                //if(pg != null)
                //    MessageBox.Show("Error in Period Generation. Message from selected Plug In : " + pg.Message);
                //else
                //    MessageBox.Show("Error in Period Generation." + e.Message);

                //return;
            //}
            AddRunTimeMessage("Period Generator", pg.Message);

            //Generate Execution Times
            UpdateStatus("Calling Execution Time Generator ...");
            IExecutionTimeGenerator etg = (IExecutionTimeGenerator) _internalConfig.selectedExecutionTimeGenerator.objMain;
            try
            {
                executionTimes = etg.GetExecutionTimes(config);
            }
            catch (Exception e)
            {
                if (etg != null)
                    MessageBox.Show("Error in Execution Time Generation. Message from selected Plug In : " + etg.Message);
                else
                    MessageBox.Show("Error in Execution Time Generation." + e.Message);

                return;
            }
            AddRunTimeMessage("Execution Time Generator", etg.Message);

            //Generate Release Offset
            UpdateStatus("Calling Release Offset Generator ...");
            IReleaseOffsetGenerator rog = (IReleaseOffsetGenerator)_internalConfig.selectedReleaseOffsetGenerator.objMain;
            try
            {
                releaseOffsets = rog.GetReleaseOffsets(config);
            }
            catch (Exception e)
            {
                if (rog != null)
                    MessageBox.Show("Error in Release Offset Generation. Message from selected Plug In : " + rog.Message);
                else
                    MessageBox.Show("Error in Release Offset Generation." + e.Message);

                return;
            }
            AddRunTimeMessage("Release Offsets", rog.Message);

            
            //Create Callback Function
            UpdateStatus("Setting Callback Function ...");
            CCallBack cb = new CCallBack();
            cb.funcReceivedTaskSet += CheckSchedulability;


            //Call Task Set Generator
            UpdateStatus("Calling Task Generator...");
            ITaskSetGenerator tsg = (ITaskSetGenerator)_internalConfig.selectedTaskSetGenerator.objMain;
            try
            {
                tsg.GenerateTaskSets(periodSets, executionTimes,releaseOffsets,  config, cb);
            }
            catch(Exception e)
            {
                if (tsg != null)
                    MessageBox.Show("Error in Task Set Generation. Message from Selected Plug In : " + tsg.Message);
                else
                    MessageBox.Show("Error in Task Set Generation." + e.Message);

                return;
            }
            AddRunTimeMessage("Task Set Generator" , tsg.Message);

            //Write Task Set Files
            UpdateStatus("Writing Task Files...");

            if (checkBoxPruneFolder.Checked)
            { //Delete Files
                DeleteFilesFromFolder(to.folderPathforFiles);
            }

            IFileFormatter ff = (IFileFormatter)_internalConfig.selectedFileFormatter.objMain;

            try
            {
                bStatus = ff.WriteFiles(_allTaskSets, to.folderPathforFiles, config);
            }
            catch (Exception e)
            {
                if (tsg != null)
                    MessageBox.Show("Error in Task Set Generation. Message from Selected Plug In : " + tsg.Message);
                else
                    MessageBox.Show("Error in Task Set Generation." + e.Message);

                return;
            }
            AddRunTimeMessage("File Formatter", ff.Message);

            UpdateStatus("Task Set Generation Complete");

        }
        
        /// <summary>
        /// Delete all files from Folder sFolderName
        /// </summary>
        /// <param name="sFolderName"></param>
        private void DeleteFilesFromFolder(string sFolderName)
        {
            string[] sFiles = Directory.GetFiles(sFolderName);
            foreach (string file in sFiles)
                File.Delete(file);
        }

        private void InitTestsAndConditions(CThreadObject to)
        {
            

            //Add Sched Tests
            UpdateStatus("Adding Schedulability Tests ...");            
            foreach (CListBoxWrapper oWrap in checkedListBoxSchedulabilityTests.CheckedItems)
                _internalConfig.selectedSchedulabilityTestWrappers.Add(oWrap);


            //Add Satisfiability Tests
            UpdateStatus("Adding Satisfiability Tests ...");
            foreach (CListBoxWrapper oWrap in checkedListBoxSatisfiabilityConditions.CheckedItems)
                _internalConfig.selectedSatisfiabilityConditionWrappers.Add(oWrap);

            _schedTests = _internalConfig.SchedulabilityTests;
            _satConditions = _internalConfig.SatisfiabilityConditions ;

            Application.DoEvents();
        }

        private bool CheckSchedulability(CTaskSet taskSet)
        {
            try
            {
                AddRunTimeMessage("Schedulability Check", "Verifying Schedulability of Task Set : " + _taskCount.ToString());

                //Add Sched and Satisfiability Tests


                if (!CheckSchedulabililityTests(_schedTests, taskSet, _config)) return false;
                if (!CheckSatisfiabilityTests(_satConditions, taskSet, _config)) return false;

                //Add Task Set to Master List
                _allTaskSets.Add(taskSet);

                UpdateStatus("Generated task set : " + _taskCount++.ToString());
                return true;
            }
            catch(Exception e)
            {
                AddRunTimeMessage("Schedulability Check", "Error: + " + e.Message);
                //Stop Execution
                return false;
            }

        }


        /// <summary>
        /// Caution: This functiom should be called only from main thread
        /// </summary>
        /// <param name="combinedMessage"></param>
        private void AddRunTimeMessage(string combinedMessage)
        {
            textBoxMessages.Text = textBoxMessages.Text + combinedMessage + "\r\n";

            //Scroll to Last Message
            textBoxMessages.SelectionStart = textBoxMessages.Text.Length;
            textBoxMessages.ScrollToCaret();
        }

        private void AddRunTimeMessage(string source, string message)
        {
            
            if (message == null) return;
            if (message.Trim() == "") return;
            string sCombinedMessage = source + " : " + message;


            if (textBoxMessages.InvokeRequired)
            {
                textBoxMessages.Invoke(new AddRunTimeMessageDelegate(this.AddRunTimeMessage), sCombinedMessage);
            }
            else
            {
                AddRunTimeMessage(sCombinedMessage);
            }
        }



        private bool CheckSchedulabililityTests(List<ISchedulabilityTest> schedTests,
                                                CTaskSet taskSet, CConfiguration config)
        {
            foreach (ISchedulabilityTest test in schedTests)
            {
                try
                {
                    if (!test.IsSatisfied(taskSet, config)) return false;
                }
                catch (Exception e)
                {
                    AddRunTimeMessage("Schedulability Tests", "Error in " + test.Description + ", Message: " + e.Message);
                    btnStop_Click(null, null);
                }

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

        /// <summary>
        /// Run the Simulation from already generated Tasks
        /// </summary>
        private void AutoStartSimulation()
        {
            UpdateStatus("Starting Simulation Run");
            fSimulation formSimulation = new fSimulation();
            formSimulation.DisplayForm(textFolderName.Text, true, _internalConfig,_config);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void setPeriodGeneratorToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DisplayGenericOptionForm(GenericOptions.PeriodGenerator);
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="genericOption"></param>
        private void DisplayGenericOptionForm(GenericOptions genericOption)
        {
            fGenericSelection oFormGS  = new fGenericSelection();
            List<CListBoxWrapper> allSelected;

            switch (genericOption)
            {
                case GenericOptions.ExecutionModel:
                    allSelected = oFormGS.DisplayForm("Select Execution Model", typeof(IExecutionModel),
                            _internalConfig.selectedExecutionModel,false);
                    if (allSelected.Count > 0)
                        _internalConfig.selectedExecutionModel =  allSelected[0];
                    break;

                case GenericOptions.ExecutionTimeGenerator:
                    allSelected=oFormGS.DisplayForm("Select Execution Time Generator", 
                                    typeof(IExecutionTimeGenerator), _internalConfig.selectedExecutionTimeGenerator
                                    ,false);

                    if (allSelected.Count > 0)
                        _internalConfig.selectedExecutionTimeGenerator =allSelected[0];
                    break;
                    

                case GenericOptions.PeriodGenerator:
                    allSelected=oFormGS.DisplayForm("Select Period Generator", typeof(IPeriodGenerator),
                                                    _internalConfig.selectedPeriodGenerator,false);
                    if (allSelected.Count > 0)
                        _internalConfig.selectedPeriodGenerator =  allSelected[0];
                    break;
                    

                case GenericOptions.TaskSetGenerator:
                    allSelected=oFormGS.DisplayForm("Select Task Set Generator", typeof(ITaskSetGenerator),
                                                        _internalConfig.selectedTaskSetGenerator,false);
                    if (allSelected.Count > 0)
                        _internalConfig.selectedTaskSetGenerator =  allSelected[0]; 
                    break;

                case GenericOptions.ReleaseOffsetGenerator:
                    allSelected=oFormGS.DisplayForm("Select Release Offset Generator", typeof(IReleaseOffsetGenerator),
                                                        _internalConfig.selectedReleaseOffsetGenerator,false);
                    if (allSelected.Count > 0)
                        _internalConfig.selectedReleaseOffsetGenerator = allSelected[0];   
                    break;

                case GenericOptions.ExecutionTraceWriter:
                    allSelected = oFormGS.DisplayForm("Select Execution Trace Writer", typeof(IDiscreteExecutionTraceWriter),
                            _internalConfig.selectedExecutionTraceWriter, false);
                    if (allSelected.Count > 0)
                        _internalConfig.selectedExecutionTraceWriter = allSelected[0];
                    break;

                default:
                    break;
            }
        }

        private void aboutToolStripMenuItem_Click(object sender, EventArgs e)
        {
            fAbout fa = new fAbout();
            fa.ShowDialog();
        }

        private void simulationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            
            
            fSimulation formSimulation = new fSimulation();
            SetTestConfiguration();
            formSimulation.DisplayForm(textFolderName.Text , false,_internalConfig, _config);
        }

        private void setExecutionTimeGeneratorToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DisplayGenericOptionForm(GenericOptions.ExecutionTimeGenerator);
        }

        private void setExecutionModelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DisplayGenericOptionForm(GenericOptions.ExecutionModel);
        }

        private void setTaskSetGeneratorToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DisplayGenericOptionForm(GenericOptions.TaskSetGenerator);
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            DialogResult dr= MessageBox.Show("Are you sure you want to stop Task Generation ?", "Please Confirm", MessageBoxButtons.YesNo);
            
            if(dr==DialogResult.Yes)
            {
                UpdateStatus("Aborting Active Thread");
                _forceStop = true;
                activeThread.Abort();
                
            }
        }

        private void SetActiveUI(bool flag)
        {
            btnStart.Enabled = flag;

        }


        private void loadConfigurationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBox.Show("This feature in under development");
        }

        private void saveConfigurationToolStripMenuItem_Click(object sender, EventArgs e)
        {
            MessageBox.Show("This feature in under development");
        }

        private void setRelToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DisplayGenericOptionForm(GenericOptions.ReleaseOffsetGenerator);
        }

        private void setExecutionTraceWriterToolStripMenuItem_Click(object sender, EventArgs e)
        {
            DisplayGenericOptionForm(GenericOptions.ExecutionTraceWriter);
        }

      

       

    }
}
