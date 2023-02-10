using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IReleaseOffsetGenerator:IRootInterface
    {
        List<double> GetReleaseOffsets(CConfiguration configuration);
        
        
    }
}
