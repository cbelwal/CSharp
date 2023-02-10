using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CSimpleFileFormatter:IFileFormatter 
    {
        private string _message;
        
        public string Description
        {
            get { return "Comma Separated Format (CSV) File, in RM Normal Priority Order (Task 0 has highest priority)"; }
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
        /// 
        /// </summary>
        /// <param name="taskSetArray"></param>
        /// <param name="FolderPath"></param>
        /// <returns></returns>
        public bool WriteFiles(List<CTaskSet> taskSetArray, string FolderPath, CConfiguration config)
        {
            int idx = 1;
            string sFullFileName;
            string sComment;

            sComment = "Period Range : " + config.MinPeriod + " - " + config.MaxPeriod;

            try
            {
                foreach (CTaskSet taskSet in taskSetArray)
                {
                    taskSet.ChangePriorityOrderToRM();
                    sFullFileName = FolderPath + "\\" + idx++.ToString().Trim() + ".txt";
                    WriteToFile(sFullFileName, sComment, taskSet);
                }
            }
            catch(Exception e)
            {
                 _message = e.Message;
                return false;

            }

            return true;

        }


        public CTaskSet ParseTaskSet(string FileContents)
        {
            string[] mainArray;
            string[] sa;
            string s;
            mainArray = FileContents.Split('\n');
            CTaskSet ts = new CTaskSet();
            
            foreach (string sTmp in mainArray)
            {
                s = sTmp.Replace('\r',' ');
                //Format of String : Event ID, priority, S, P, C
                if (!s.StartsWith("//") && s.Trim() != "")
                {
                    sa = s.Split(',');
                    CTask T = new CTask(sa[0],
                                        Convert.ToInt32(sa[1]),
                                        Convert.ToDouble(sa[3]),
                                        Convert.ToDouble(sa[4]),
                                        Convert.ToDouble(sa[2]),
                                        0);
                   
                    ts.Add(T); //4 has a higher priority than 1
                }

            }
            return ts;
        }

        /// <summary>
        /// Write Files
        /// </summary>
        /// <param name="sFullFileName"></param>
        /// <param name="taskSet"></param>
        private void WriteToFile(string sFullFileName, string sComment,  CTaskSet taskSet)
        {
            string sMain = "";
            string sTmp = "";

            CTask [] array = taskSet.GetAllTasksInPriorityOrder();

            sMain = "//" + sComment + "\r\n";
            sMain = sMain + "//Hyperperiod: " + taskSet.HyperPeriod + ", Utilization: " + taskSet.Utilization + "\r\n";
            sMain = sMain + "//ID,Priority,Offset,Period, Execution Time" + "\r\n";
            

            foreach (CTask t in array)
            {
                sTmp = t.ID + "," + t.iPriority.ToString() + ","
                            + t.Offset.ToString() + "," + t.Period.ToString() + "," 
                            + t.ExecutionTime.ToString();

                sMain = sMain + sTmp + "\r\n";
            }

            writeToFile(sFullFileName, sMain);

        }

        /// <summary>
        /// Write To File
        /// </summary>
        /// <param name="sFileName"></param>
        /// <param name="text"></param>
        /// <returns></returns>
        private bool writeToFile(string sFileName, string text)
        {
            TextWriter tw = new StreamWriter(sFileName, false);
            tw.WriteLine(text);
            tw.Close();
            tw.Dispose();
            return true;
        }
    }
}
