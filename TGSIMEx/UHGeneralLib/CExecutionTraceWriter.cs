using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CExecutionTraceWriter:IDiscreteExecutionTraceWriter
    {
        private string _message;

        public CExecutionTraceWriter()
        { }


        public bool WriteFile(CTaskExecutionTrace executionTrace,
                string FolderPath, string TaskSetID, CConfiguration config)
        {
            int ii;
            string sMain="";
            CTask task;
            string sFullFileName = FolderPath + "\\" +  TaskSetID + ".txt";

            _message = "";
            
            try
            {
                for (ii = 0; ii < executionTrace.Count; ii++)
                {
                    task = executionTrace.Get(ii);
                    if(task != null)
                        sMain = sMain + ii.ToString() + "\t" + task.ID;
                    else //Write blank if no task ID
                        sMain = sMain + ii.ToString() + "\t" + " ";

                    sMain = sMain + "\r\n";
                }
                writeToFile(sFullFileName, sMain);
                return true;
            }
            catch(Exception e)
            {
                _message = e.Message;
                return false;
            }
        }


        //Return Task Set after loading from
        //File is formatted like
        //0     T1
        //1     T1
        //2     T2
        //...
        public CTaskExecutionTrace ParseTaskExecutionTrace(string 
                                            FileContents)
        {
            string[] sArray,sArray_1;
            
            long time;
            CTask task;
            CTaskExecutionTrace et = new CTaskExecutionTrace();

            try
            {
                sArray = FileContents.Split('\n');

                foreach (string s in sArray)
                {
                    sArray_1 = s.Split('\t');
                    time = Convert.ToInt32(sArray_1[0]);
                    task = new CTask(sArray_1[1], 0, 0, 0, 0, 0);
                    et.Add(time, task);
                }
                return et;
            }
            catch (Exception e)
            {
                _message = e.Message;
                return null;
            }
        }


        public string Description
        {
            get { return "Simple Time:Task ID Trace Writer"; }
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
