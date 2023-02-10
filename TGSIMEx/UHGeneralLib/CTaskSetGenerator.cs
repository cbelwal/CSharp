using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;


namespace UHGeneralLib
{
    public class CTaskSetGenerator : ITaskSetGenerator
    {
        private string _message;
        public void GenerateTaskSets(List<CPeriodSet> periodSets, 
                                     List<double> executionTimes,
                                     List<double> releaseOffsets,
                                     CConfiguration config, 
                                     CCallBack addTaskCallback)
        {
            CTaskSet ts;
            CTask task;
            int ii;
            int executionTimeIdx=0;
            int releaseOffsetIdx=0;
            int count = 0;
            


            _message = "";

            if (periodSets.Count < config.NumberOfTaskSets)
                _message = "Number of period sets is lower than number of required tasks. This task generator can only generate number of tasks same as period sets";

            
            periodSets.Sort();

            try
            {
                foreach (CPeriodSet ps in periodSets)
                {
                    ts = new CTaskSet();

                    if (count == config.NumberOfTaskSets) break;

                    for (ii = 0; ii < ps.Count; ii++)
                    {
                        task = new CTask("T" + ii.ToString(), ii, ps.Get(ii),
                                            executionTimes[executionTimeIdx++],
                                            releaseOffsets[releaseOffsetIdx], 0);

                        ts.Add(task);

                        if (executionTimeIdx >= executionTimes.Count) executionTimeIdx = 0;
                        if (releaseOffsetIdx >= releaseOffsets.Count) releaseOffsetIdx = 0;
                    }

                    //Task Set Created, Now Call Delegate
                    if(addTaskCallback.AddGeneratedTaskSet(ts)) count++;
                }
            }
            catch (Exception e)
            {
                _message = "Error in Task Generation Routine : " + e.Message;
            }
        }


        public string Description
        {
            get { return "Simple Task Set Generator by Task Execution Times and Release Offsets"; }
        }

        public string Developer
        {
            get { return "RTSL - Univ. of Houston"; }
        }

        public string Message
        {
            get { return _message; }
        }

    }
}
