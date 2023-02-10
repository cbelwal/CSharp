using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IExecutionTimeGenerator : IRootInterface
    {
        List<double> GetExecutionTimes(CConfiguration configuration);
        
    }
}
