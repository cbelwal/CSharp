using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace UHGeneralLib
{
    public class CPreemptiveExecutionModel:IExecutionModel
    {
        private CTaskExecutionTrace _executionTrace;
        private int _stepsTotal;
        private int _interferenceCount;
        private string _message;
        private long _lastDiscreteTimeValue;

        public CTaskExecutionTrace ExecutionTrace
        {
            get { return _executionTrace; }
        }

        public int InterferenceCount
        {
            get { return _interferenceCount; }
        }

        public int ComputationTime
        {
            get { return _stepsTotal; }
        }

        public int UnschedulableTaskID
        {
            get { return 0; }
        }


        public List<Object> OtherInfo
        {
            get { return null; }
        }

        public string Description
        {
            get { return "Preemptive Execution Model"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }

        public string Message
        {
            get { return _message; }
        }

        public long LastDiscreteTimeValue
        {
            get { return _lastDiscreteTimeValue; }
        }

        /// <summary>
        /// Run simulation of the task set till the Hyper-period
        /// </summary>
        /// <param name="taskSet"></param>
        /// <returns></returns>
        public bool SimulateExecution(CTaskSet taskSet)
        {
            long maxTime = (long)taskSet.HyperPeriod;
            return SimulateExecution(taskSet, maxTime);
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="taskSet"></param>
        /// <param name="maxTime"></param>
        /// <returns></returns>
        public bool SimulateExecution(CTaskSet taskSet, long maxTime)
        {
            _executionTrace = new CTaskExecutionTrace();
            _message = "";
            SortedList allTasks = new SortedList();
            

            //Reverse Priority Order in Task Set
            taskSet.ReversePriorityOrder();

            //Convert Task Set to SortedList
            CTask[] array = taskSet.GetAllTasksInPriorityOrder();

            if (maxTime <= 0)
            {
                _message = "Time to run simulation is not correct";
                return false;
            }

            foreach (CTask t in array)
                allTasks.Add(t.iPriority, t);

            //Get Reverse Priority Tasks
            //allTasksReverseOrder = ReversePriorities(allTasks);
            try
            {
                if (findResponseTime(allTasks, 0, maxTime) < 0)
                {
                    return false;
                }
                else
                {
                    return true;
                }
            }
            catch (Exception e)
            {
                _message = e.Message;
                return false;
            }
        }

      

        /// <summary>
        /// 
        /// </summary>
        /// <param name="L"></param>
        /// <returns></returns>
        private long findResponseTime(SortedList eventSet, int priorityOfEvent, long L)
        {
            ArrayList Q = new ArrayList();
            SortedList G = new SortedList();

            //Reverse Priority Order, Such that task with lowest Priority is at index 0
            long t;
            int ii;
            int Gap = 0;
            int Ci = 0;
            int retVal, lowestPriorityIndex;
            CTask hPriority, tmpE;
            int stepsInternal;

            long currentTime;
            int noOfInterferencesLowestPriority;


            if (priorityOfEvent > 0)
                lowestPriorityIndex = priorityOfEvent;
            else
                lowestPriorityIndex = 0;



            //Add all higher priority events 
            for (ii = getIndexofHighestPriority(eventSet); ii > getIndex(lowestPriorityIndex); ii--)
            {
                tmpE = (CTask)eventSet[ii];
                tmpE.dExecutedSteps = 0; //Important to do this
                G.Add(tmpE.iPriority, tmpE);
            }

            if (priorityOfEvent > 0) //Lowest priority can be 1
            {
                tmpE = (CTask)eventSet[priorityOfEvent];
                Ci = (int)tmpE.ExecutionTime;
            }
            else
            {
                Ci = -1;
            }


            _stepsTotal = 0;
            noOfInterferencesLowestPriority = 0;
            _interferenceCount = 0;


            for (t = 0; t < L; t++)
            {
                _lastDiscreteTimeValue = t;
                System.Windows.Forms.Application.DoEvents();


                currentTime = t; //Maintain Current Time
                retVal = putEventInQ(t, G, Q);
                if (retVal < 0) return retVal;

                _stepsTotal++;

                //Execute task in Queue
                if (Q.Count > 0)
                {
                    hPriority = (CTask)Q[Q.Count - 1]; // Last task
                    stepsInternal = 0;
                    foreach (CTask T in Q)
                    {
                        stepsInternal++;
                        if (T.iPriority >= hPriority.iPriority)
                            hPriority = T;
                    }
                    if (stepsInternal == 0) _stepsTotal++;
                    else _stepsTotal += stepsInternal;

                    if (Gap > 0)
                    {
                        noOfInterferencesLowestPriority++;
                        _interferenceCount++;
                    }
                    Gap = 0; 

                    //Execute task hPriority
                    hPriority.dExecutedSteps++;
                    if (hPriority.dExecutedSteps == hPriority.ExecutionTime)
                    {
                        Q.Remove(hPriority);
                        hPriority.dExecutedSteps = 0;
                    }

                    
                    
                    _executionTrace.Add(t, hPriority);

                    //Reset all other executed steps to 0
                    stepsInternal = 0;
                    foreach (CTask T in Q)
                    {
                        stepsInternal++;
                        if (T.iPriority != hPriority.iPriority)
                        {
                            if (T.dExecutedSteps > 0) _interferenceCount++;
                            //T.dExecutedSteps = 0; //Dont do this for Preemptive


                        }
                    }
                    if (stepsInternal == 0) _stepsTotal++;
                    else _stepsTotal += stepsInternal;

                }
                else
                {
                    Gap++;
                    _executionTrace.Add(t, null);
                    if (Gap == Ci)// && t >= Ci * G.Count) //Gap found
                        return (t + 1); //If Ci is -ve this will never be reached
                }


            }
            if (priorityOfEvent > 0) return -1; //No gap was found
            else return 0; // All events were schedulable
        }


        /// <summary>
        /// Scans through the task list and sees if any tasks should be in Queue
        /// </summary>
        private int putEventInQ(long t, SortedList G, ArrayList Q)
        {
            int stepsInternal = 0;
            //Here the sorting order of tasks is not necessary hence we have used G.Values
            foreach (CTask T in G.Values) //Tasks will come in Sorted Order
            {
                stepsInternal = 0;
                for (int et = (int)T.Offset; et <= t; et = et + (int)T.Period)
                {
                    stepsInternal++;
                    if (et == t)
                    {
                        if (presentInQ(T, Q))
                            return -T.iPriority; //Some Event has not completed before next one has arrived -> Deadline Miss
                        Q.Add(T);
                        break;
                    }
                }
                if (stepsInternal == 0) _stepsTotal++;
                else _stepsTotal += stepsInternal;

            }
            return 0;
        }

        private bool presentInQ(CTask E, ArrayList Q)
        {
            foreach (CTask T in Q)
            {
                _stepsTotal++;
                if (T.iPriority == E.iPriority) return true;
            }
            return false;
        }

        private int getIndexofHighestPriority(SortedList list)
        {
            return list.Count;
        }

        private int getIndex(int eventPriority)
        {
            return eventPriority;
        }
        
    }
}
