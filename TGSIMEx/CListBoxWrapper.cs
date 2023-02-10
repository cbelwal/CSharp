using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using TGSIMExLib;

namespace TaskGen
{
    class CListBoxWrapper
    {
        public IRootInterface objMain;
        public int index;

        public override string ToString()
        {
            return objMain.Description + "  ( Source: " + objMain.Developer + " )";
        }

    }
}
