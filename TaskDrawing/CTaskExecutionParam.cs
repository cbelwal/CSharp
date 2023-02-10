using System;
using System.Collections;
using System.Linq;
using System.Text;

namespace TaskDrawing
{
    //This Exists for Each Discrete Time

    public class CTaskExecutionParam
    {
        public string taskID;
        public bool bEnd;
        public ArrayList releasingTasks; 

        /// <summary>
        /// Tasks is in the format <T1;T2;T3>
        /// </summary>
        /// <param name="tasks"></param>
        /// <param name="time"></param>
        public void StoreReleasingTasks(string tasks, string taskPrefix)
        {
            string[] sarray;
            string taskID;
            tasks = tasks.Replace("<", "");
            tasks = tasks.Replace(">", "");

            sarray = tasks.Split(';');

            if (sarray.Length > 0)
            {
                releasingTasks = new ArrayList();

                foreach (string s in sarray)
                {
                    taskID = taskPrefix + s;
                    if(s.Trim() != "")
                        releasingTasks.Add(taskID);
                }
               
            }

        }

    }
}
