using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IExecutionModel:IRootInterface
    {
        CTaskExecutionTrace ExecutionTrace { get; }
        int InterferenceCount { get; }
        int ComputationTime { get; }
        int UnschedulableTaskID { get; }
        long LastDiscreteTimeValue { get; }
        List<Object> OtherInfo { get; }

        bool SimulateExecution(CTaskSet taskSet, long maxTime);

        //Overloaded function automatically runs till the Hyper-Period
        bool SimulateExecution(CTaskSet taskSet);
       
    }
}
