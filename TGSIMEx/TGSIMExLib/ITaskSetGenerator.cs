using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface ITaskSetGenerator:IRootInterface
    {
        void GenerateTaskSets(List<CPeriodSet> periodSets, 
                              List<double> executionTimes,
                              List<double> releaseOffsets,
                              CConfiguration config, 
                              CCallBack addTaskCallback);   

    }
}
