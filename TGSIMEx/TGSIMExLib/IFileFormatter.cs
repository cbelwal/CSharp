using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    public interface IFileFormatter : IRootInterface
    {
        CTaskSet ParseTaskSet(string FileContents);
        bool WriteFiles(List<CTaskSet> taskSets, string sFolderPath, CConfiguration config);
   
    }
}
