using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IPrimeNumberGenerator : IRootInterface
    {
        List<double> GeneratePrimeNumbers(double min, double max);
        

    }
}
