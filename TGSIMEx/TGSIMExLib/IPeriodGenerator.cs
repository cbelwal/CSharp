using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IPeriodGenerator : IRootInterface
    {
         List<CPeriodSet> GetPeriods(CConfiguration configuration, IPrimeNumberGenerator primeGenerator);
         long GetMaximumUniquePeriodSets(CConfiguration configuration);
         
    }
}
