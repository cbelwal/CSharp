using System;
using System.Collections;
using System.Linq;
using System.Text;
using System.IO;

namespace TaskDrawing
{
    public class CTaskExecutionTrace
    {

        public Hashtable hshTimeTrace; //Contains Task Execuiton Params
        public Hashtable hshTaskID;
        public int maxTime;
        

        /// <summary>
        /// Reads from an Execution Trace file and creates execution trace data
        /// T1,1,0
        /// T1,0,1
        /// T2,1,0
        /// </summary>
        /// <param name="sFilePath"></param>
        public void ReadTaskExecutionTraceFile(string sFilePath, string taskPrefix)
        {
            ArrayList file = readFromFile(sFilePath);
            string [] sarray;
            string line;
            CTaskExecutionParam ep;
            int time = 0;

            hshTimeTrace = new Hashtable();
            hshTaskID = new Hashtable();

            foreach (string l in file)
            {
                //Replace Tab with ,

                if (l.Trim() != "")
                {
                    

                    line = l.Replace('\t', ',');
                    line = line.Trim();

                    if (line.StartsWith("Tasks")) //Ignore Comments Line
                    {
                        //Add all Task ID to unique HashTable
                        AddTaskIDstoHaskTable(line, taskPrefix);
                    }
                    else if (!line.StartsWith("//")) //Ignore Comments Line
                    {
                       // if(line.StartsWith(",")) // No Task Defined
                        
                        
                        sarray = line.Split(',');
                        ep = new CTaskExecutionParam();
                        ep.taskID = taskPrefix + sarray[0];

                        if (sarray[1] == "1")
                            ep.bEnd = true;
                        else
                            ep.bEnd = false;

                        //Now find Start Times
                        if (sarray.Length > 2)
                            ep.StoreReleasingTasks(sarray[2], taskPrefix);

                        hshTimeTrace.Add(time++, ep);
                    }
                }
            }
            maxTime = time - 1;
        }


        

        /// <summary>
        /// Takes input of the form Tasks,1,2,3
        /// </summary>
        /// <param name="taskID"></param>
        private void AddTaskIDstoHaskTable(string tasksLine, string taskPrefix)
        {
            string[] sarray;
            int ii;
            string taskID;
            sarray = tasksLine.Split(',');

            for(ii=1;ii<sarray.Length;ii++)
            {
                taskID = taskPrefix + sarray[ii];
                
                if (!hshTaskID.ContainsKey(taskID))
                    hshTaskID.Add(taskID, hshTaskID.Count);

            }

        }


        private ArrayList readFromFile(string filePath)
        {
            ArrayList list = new ArrayList();
            try
            {
                StreamReader reader = new StreamReader(filePath);
                while (!reader.EndOfStream)
                {
                    string str = reader.ReadLine().Trim();
                    list.Add(str);
                }
                reader.Close();
                reader.Dispose();
            }
            catch
            {
                list.Clear();
            }
            return list;
        }
    }
}
