using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IDiscreteExecutionTraceWriter:IRootInterface
    {
        bool WriteFile(CTaskExecutionTrace executionTrace,
                string FolderPath, string TaskSetID, CConfiguration config);
        CTaskExecutionTrace ParseTaskExecutionTrace(string FileContents);
    }
}
