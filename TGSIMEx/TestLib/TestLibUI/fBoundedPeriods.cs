using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using UHPeriodLib;
using TGSIMExLib;

namespace TestLibUI
{
    public partial class fBoundedPeriods : Form
    {
        public fBoundedPeriods()
        {
            InitializeComponent();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnStart_Click(object sender, EventArgs e)
        {
            //Test for 
            //List<List<double>> c;
            //List<double> main = new List<double>(); 
            //main.Add(0); main.Add(1); main.Add(2); main.Add(3); main.Add(4);
            //CCombinationGenerator comb = new CCombinationGenerator();
            //c = comb.GenerateCombinations(main, 3);
            //return;
            
            
            double maxLCM,LCM;
            int noOfTasks;
            List<CPeriodSet> periods;

            if (textTasksToGen.Text.Trim() != "")
                noOfTasks = Convert.ToInt32(textTasksToGen.Text);
            else
                noOfTasks = 100;



            CConfiguration oConfig =  CConfiguration.GetObject(noOfTasks,
                                                        Convert.ToInt32(textTasksPerSet.Text),
                                                        Convert.ToDouble(textMinTaskPeriod.Text),
                                                        Convert.ToDouble(textMaximumTaskPeriod.Text),
                                                        1,10,0,0);
            IPeriodGenerator oPeriodGen=null;

            //Select Period Generator
            switch(comboPeriodMethod.SelectedIndex)
            {
                case 0:
                    oPeriodGen = new CBoundedPeriods();
                    break;

                case 1:
                    oPeriodGen = new CRandomPeriods();
                    break;

                case 2:
                    oPeriodGen = new CAllPeriods();
                    break;

                default:
                    break;
            }

            
            
            //Determine maximum possible periods
            UpdateStatus("Computing Number of Period Sets ...");
            textNumberOfTaskSets.Text = oPeriodGen.GetMaximumUniquePeriodSets(oConfig).ToString();

            textNumberOfPossibleTaskSets.Text = GetMaximumPossibleSets(oConfig).ToString();

            UpdateStatus("Generating Period Sets ...");

            periods = oPeriodGen.GetPeriods(oConfig, new CSieveOfEratosthenes());

            listMain.Items.Clear();
            listMain.Items.Add( "<Periods>, LCM");

            UpdateStatus("Displaying Period Sets ...");
            maxLCM = 0;

            foreach (CPeriodSet period in periods)
            {
                Application.DoEvents();
                LCM = period.GetLCM();

                if (LCM > maxLCM) maxLCM = LCM;
                //listMain.Items.Add(period.GetPeriodString() + "," + LCM.ToString());
                listMain.Items.Add(LCM.ToString());
                if (listMain.Items.Count == oConfig.NumberOfTaskSets) break;
            }
            textLCM.Text = maxLCM.ToString();


            UpdateStatus("Ready");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sMessage"></param>
        private void UpdateStatus(string sMessage)
        {
            toolStripStatusLabel1.Text = sMessage;
            Application.DoEvents();
        }

        private double GetMaximumPossibleSets(CConfiguration config)
        {
            double sum;
            long delta;
            long MxSize;
            CCombinationGenerator comb = new CCombinationGenerator();
            CSieveOfEratosthenes sieve = new CSieveOfEratosthenes();


            delta = (long)(config.MaxPeriod - config.MinPeriod + 1);

            List<double> primes = sieve.GeneratePrimeNumbers(0.0, config.MaxPeriod);

            sum = 0;
            foreach (double prime in primes)
            {
                if (prime <= delta / 2)
                {
                    MxSize = (long)(delta / prime + 1);
                    sum += comb.GetNumberOfCombinations(MxSize, config.NumberOfTasksPerSet);
                }
            }

            return sum;
        }

       
        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void btnClipboard_Click(object sender, EventArgs e)
        {
            string sMain = "";
            string sTmp = "";
            int idx=0;
            
            foreach (string s in listMain.Items)
            {
                UpdateStatus("Copying to clipboard : " + idx++.ToString());
                sTmp = s.Replace(',', '\t');
                sMain = sMain + sTmp + "\r\n";
            }
            Clipboard.SetData(DataFormats.Text, sMain);
            UpdateStatus("Ready");
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void fBoundedPeriods_Load(object sender, EventArgs e)
        {
            comboPeriodMethod.SelectedIndex = 0;
        }

    }
}
