using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public class CTaskSet
    {
        SortedList<string,CTask> listMain;

        public CTaskSet()
        {
            listMain = new SortedList<string, CTask>();
        }

        public double Utilization
        {
            get
            {
                double U = 0;
                foreach (CTask task in listMain.Values)
                {
                    U = U + task.Utilization;
                }
                return U;
            }
        }

        public int Count
        {
            get
            {
                return listMain.Count;
            }
        }

        public double HyperPeriod
        {
            get
            {
                List<double> periods = new List<double>();

                foreach (CTask task in listMain.Values)
                    periods.Add(task.Period);
                
                return CMath.LCM(periods);
            }

        }



        public bool Add(CTask task)
        {
            try
            {
                listMain.Add(task.ID, task);
                return true;
            }
            catch
            {
                return false;
            }
        }

        public bool Delete(CTask task)
        {
            try
            {
                listMain.Remove(task.ID);
                
                return true;
            }
            catch
            {
                return false;
            }
        }


        public bool isPresent(CTask task)
        {
            if (listMain.ContainsValue(task)) return true;
            else return false;
        }

        /// <summary>
        /// Returns all Tasks in the Priority Order, with Task at index 0 having lowest priority
        /// </summary>
        /// <returns></returns>
        public CTask[] GetAllTasksInPriorityOrder()
        {
            SortedList tmpSortedList = new SortedList();
            int idx = 0;
            CTask [] array = new CTask[listMain.Count];
            foreach (CTask task in listMain.Values)
                tmpSortedList.Add(task.iPriority, task);


            foreach(CTask task in tmpSortedList.Values)
                array[idx++]=task;

            return array;
        }


        /// <summary>
        /// Reverses the Priority Order of Tasks such that task with lowest priority is assigned value 1
        /// and task with highest priority is assigned value N
        /// </summary>
        /// <returns></returns>
        public bool ReversePriorityOrder()
        {
            SortedList tmpSortedList = new SortedList();
            
            int N = listMain.Count;
            try
            {
                foreach (CTask task in listMain.Values)
                    tmpSortedList.Add(task.Period, task);

                listMain.Clear();

                foreach (CTask task in tmpSortedList.Values)
                {
                    //Let N = 5
                    //if iPriority =0, then new iPriority = 5 - 0 = 5
                    //if iPriority =1, then new iPriority = 5 - 1 = 4
                    //if iPriority =2, then new iPriority = 5 - 2 = 3
                    //if iPriority =3, then new iPriority = 5 - 3 = 2
                    //if iPriority =4, then new iPriority = 5 - 4 = 1
                    task.iPriority = N - task.iPriority;
                    listMain.Add(task.ID, task);
                }

                return true;
            }
            catch
            {
                return false;
            }
        }

        /// <summary>
        /// Changes the priorities of Task based on Rate-Monotonic Priority Assignment
        /// Task at Index 0 has the highest arrival rate
        /// </summary>
        /// <returns></returns>
        public bool ChangePriorityOrderToRM()
        {
            SortedList tmpSortedList = new SortedList();
            int iPriority = 0;

            try
            {
                foreach (CTask task in listMain.Values)
                    tmpSortedList.Add(task.Period, task);

                listMain.Clear();

                foreach (CTask task in tmpSortedList.Values)
                {
                    task.iPriority = iPriority++;
                    listMain.Add(task.ID, task);
                }

                return true;
            }
            catch
            {
                return false;
            }
            

        }

       

    }
}
