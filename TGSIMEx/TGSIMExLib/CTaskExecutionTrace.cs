using System;
using System.Collections.Generic;
using System.Collections;
using System.Linq;
using System.Text;

namespace TGSIMExLib
{
    
    public class CTaskExecutionTrace
    {
        Hashtable _hshMain;

        public long Count
        {
            get { return _hshMain.Count; }
        }

        public CTaskExecutionTrace()
        {
            _hshMain = new Hashtable();
        }

        public bool Add(long time, CTask task)
        {
            try
            {
                _hshMain.Add(time, task);
                return true;
            }
            catch
            {
                return false;
            }

        }

        public bool Clear()
        {
            _hshMain.Clear();
            return true;
        }


        public CTask Get(long time)
        {
            try
            {
                return (CTask) _hshMain[time];
            }
            catch
            {
                return null;
            }
        }
    }
}
