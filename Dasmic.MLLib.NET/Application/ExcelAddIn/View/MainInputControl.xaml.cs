using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using Excel = Microsoft.Office.Interop.Excel;
using ExcelAddIn.ViewModel;

namespace ExcelAddIn.View
{
    /// <summary>
    /// Interaction logic for MainSelection.xaml
    /// </summary>
    public partial class MainInputControl : UserControl
    {
        public Excel.Application ExcelApplication
        {
            set
            {
                ((VMMainInputControl)this.DataContext).ExcelApplication
                        = value;
            }
        }



        public MainInputControl()
        {
            InitializeComponent();
        }
    }
}
