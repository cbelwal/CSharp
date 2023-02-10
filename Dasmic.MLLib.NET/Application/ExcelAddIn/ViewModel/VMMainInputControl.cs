using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.ComponentModel;
using Excel = Microsoft.Office.Interop.Excel;

namespace ExcelAddIn.ViewModel
{
    public class VMMainInputControl:INotifyPropertyChanged
    {
        Excel.Application _excelApplication;

        public Excel.Application ExcelApplication
        {
            set
            {
                _excelApplication = value;               
            }
        }

        public event PropertyChangedEventHandler PropertyChanged;

        private void NotifyPropertyChanged(String propertyName = "")
        {
            if (PropertyChanged != null)
            {
                PropertyChanged(this, new PropertyChangedEventArgs(propertyName));
            }
        }


        public string CurrentRange
        {            
            get
            {
                if (_excelApplication != null)
                {
                    Excel.Range range = (Excel.Range)_excelApplication.Selection;
                    range.NumberFormat = "$#,##0_);($#,##0)";
                    return range.Address;// Text ToString();
                }
                else
                    return "";
            }
        }


        public void OnSheetChange(object Sh, Excel.Range Target)
        {
            NotifyPropertyChanged("CurrentRange");
        }

    }
}
