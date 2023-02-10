using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface ISchedulabilityTest : IRootInterface
    {
        bool IsSatisfied(CTaskSet taskSet, CConfiguration configuration);
    }
}
