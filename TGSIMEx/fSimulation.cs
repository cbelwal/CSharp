using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using System.Threading;
using TGSIMExLib;
using System.IO;

namespace TaskGen
{  
    partial class fSimulation : Form
    {
        private CInternalConfiguration _internalConfig;
        private CConfiguration _config;
        private Thread _activeThread;
        private delegate void AddTaskSetSchedulabilityDelegate(string sMessage);
        private delegate void AddDiscreteTimeExecutionDelegate(CDiscreteTimeExecutionData dted);
        private bool _forceStop;

        public fSimulation()
        {
            InitializeComponent();
            textFolderName.Text = Application.StartupPath + "\\Files";
            textTraceFolder.Text = Application.StartupPath + "\\Files";
        }

        

        private void UpdateStatus(string sMessage)
        {
            toolStripStatusLabel1.Text = sMessage.Trim();
        }


        public void DisplayForm(string sTaskFolder, bool bAutoRun, 
            CInternalConfiguration internalConfig, CConfiguration config)
        {
            _config = config;
            if (internalConfig.selectedExecutionModel == null)
            {
                MessageBox.Show("Please select execution model");
                return;
            }

            this.Text = "Run Simulation using : " + internalConfig.selectedExecutionModel.objMain.Description;
            UpdateStatus("Ready");

            textFolderName.Text = sTaskFolder;
            _internalConfig = internalConfig;

            if (bAutoRun) AsyncRunSimulation();

            //Add Code
            this.ShowDialog();
        }


        private void AsyncRunSimulation()
        {
            //Run in a New Thread
            SetActiveUI(false);
            UpdateStatus("Creating new thread ");
            Thread newThread = new Thread(RunSimulation);
            newThread.Start();
            _activeThread = newThread;

            if(!this.Visible) this.ShowDialog();
            while (newThread.IsAlive) Application.DoEvents();
            SetActiveUI(true);

            UpdateStatus("Ready");
        }

        private void SetActiveUI(bool flag)
        {
            btnStart.Enabled = flag;
        }

        private void ResetOutput()
        {
            textDiscreteTaskExecution.Text = "";
            textTaskSetSchedulability.Text = "";
        }



        private void RunSimulation()
        {
            bool bStatus;
            IExecutionModel em = (IExecutionModel)_internalConfig.selectedExecutionModel.objMain;
            IFileFormatter ff = (IFileFormatter) _internalConfig.selectedFileFormatter.objMain;
            IDiscreteExecutionTraceWriter etw = (IDiscreteExecutionTraceWriter)_internalConfig.selectedExecutionTraceWriter.objMain;

            string sFileContents;
            string sFileName;
            CTaskSet ts;

            if(textFolderName.Text.Trim() == "")
            {
                MessageBox.Show("Please select a Valid Folder");
                return;
            }

            DirectoryInfo di = new DirectoryInfo(textFolderName.Text);


            if (!di.Exists)
            {
                MessageBox.Show("Selected folder does not exists");
                return;
            }

            FileInfo[] rgFiles = di.GetFiles("*.*");

            foreach (FileInfo fi in rgFiles)
            {
                //Iterate over Task Set

                Application.DoEvents();
                UpdateStatus("Reading and parsing file " + fi.Name);
                sFileContents = ReadFromFile(fi.FullName);
                ts = ff.ParseTaskSet(sFileContents);

                UpdateStatus("Processing file " + fi.Name + " till " + ts.HyperPeriod + " steps ...");
                bStatus = em.SimulateExecution(ts);
                
                if(bStatus)
                    AddTaskSetSchedulability(fi.Name + ", TRUE" + ", Last Time: " + em.LastDiscreteTimeValue.ToString());
                else
                    AddTaskSetSchedulability(fi.Name + ", FALSE" + ", Last Time: " + em.LastDiscreteTimeValue.ToString());


                CDiscreteTimeExecutionData dted = new CDiscreteTimeExecutionData();
                dted.hp = ts.HyperPeriod;
                dted.sFileName = fi.Name;
                dted.executionTrace = em.ExecutionTrace;
                
                AddDiscreteTimeExecution(dted);

                if (textTraceFolder.Text.Trim() != "")
                {
                    //Convert 1.txt to 1_txt
                    sFileName = fi.Name.Replace('.', '_');

                    //Write to File
                    etw.WriteFile(dted.executionTrace,
                            textTraceFolder.Text.Trim(), sFileName, _config);
                }
            }
            UpdateStatus("Completed processing of all files");
        }


        /// <summary>
        /// 
        /// </summary>
        /// <param name="sMessage"></param>
        private void AddTaskSetSchedulability(string sMessage)
        {
            if (textTaskSetSchedulability.InvokeRequired)
            {
                textTaskSetSchedulability.Invoke(new AddTaskSetSchedulabilityDelegate(this.AddTaskSetSchedulability),sMessage);
            }
            else
            {
                textTaskSetSchedulability.Text = textTaskSetSchedulability.Text + sMessage + "\r\n";
                //Scroll to Last Message
                textTaskSetSchedulability.SelectionStart = textTaskSetSchedulability.Text.Length;
                textTaskSetSchedulability.ScrollToCaret();
            }

            
        }


        private void AddDiscreteTimeExecution(CDiscreteTimeExecutionData dted)
        {
            CTaskExecutionTrace executionTrace;
            int ii=0;
            CTask task;
            if (!checkBoxDisplayDiscreteTimeExecution.Checked) return;

            UpdateStatus("Displaying discrete time execution ...");

            if (textDiscreteTaskExecution.InvokeRequired)
            {
                textDiscreteTaskExecution.Invoke(new AddDiscreteTimeExecutionDelegate(this.AddDiscreteTimeExecution), dted);
            }
            else
            {
                textDiscreteTaskExecution.Text = "---->" + dted.sFileName + ", Hyperperiod = " + dted.hp.ToString() + "\r\n";
                executionTrace = dted.executionTrace;

                for(ii=0;ii<executionTrace.Count;ii++)
                {
                    task = executionTrace.Get(ii);
                    Application.DoEvents();

                    if (task != null)
                    {
                        textDiscreteTaskExecution.Text = textDiscreteTaskExecution.Text + ii.ToString() +
                                                            "\t" + task.ID + "\r\n";
                    }
                    //Scroll to Last Message
                    textDiscreteTaskExecution.SelectionStart = textDiscreteTaskExecution.Text.Length;
                    textDiscreteTaskExecution.ScrollToCaret();
                }
                
            }

        }

        private void btnStart_Click(object sender, EventArgs e)
        {
            AsyncRunSimulation();
        }


        //Reads from file and returns contents as string
        private string ReadFromFile(string sFullFilePath)
        {
            string tmpStr = "",mainStr = "";
            try
            {
                StreamReader reader = new StreamReader(sFullFilePath);

                while (!reader.EndOfStream)
                {
                    tmpStr = reader.ReadLine().Trim();
                    mainStr = mainStr + tmpStr + "\r\n";

                }

                return mainStr;
            }
            catch (Exception e)
            {
                MessageBox.Show(e.Message);
                return "";
            }
        }

        private void exitToolStripMenuItem_Click(object sender, EventArgs e)
        {
            this.Close();
        }

        private void btnCopyToClipboard_2_Click(object sender, EventArgs e)
        {
            Clipboard.SetData(DataFormats.Text, textDiscreteTaskExecution.Text);
            UpdateStatus("Data copied to clipboard");
        }

        private void btnCopyToClipboard_1_Click(object sender, EventArgs e)
        {
            Clipboard.SetData(DataFormats.Text, textTaskSetSchedulability.Text);
            UpdateStatus("Data copied to clipboard");
        }

        private void btnStop_Click(object sender, EventArgs e)
        {
            DialogResult dr = MessageBox.Show("Are you sure you want to stop the Simulation run ?", "Please Confirm", MessageBoxButtons.YesNo);

            if (dr == DialogResult.Yes)
            {
                UpdateStatus("Aborting active thread (Please wait)...");
                _forceStop = true;
                _activeThread.Abort();

            }
        }

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.SelectedPath = Application.StartupPath;
            folderBrowserDialog1.ShowDialog();
            textFolderName.Text = folderBrowserDialog1.SelectedPath;
        }

        private void btnBrowseTraceFolder_Click(object sender, EventArgs e)
        {
            folderBrowserDialog1.SelectedPath = Application.StartupPath;
            folderBrowserDialog1.ShowDialog();
            textTraceFolder.Text = folderBrowserDialog1.SelectedPath;
        }


        private void fSimulation_Load(object sender, EventArgs e)
        {

        }

    }

    class CDiscreteTimeExecutionData
    {
        public double hp;
        public string sFileName;
        public CTaskExecutionTrace executionTrace;
    }
}
