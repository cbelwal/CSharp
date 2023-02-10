using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml.Linq;
using Excel = Microsoft.Office.Interop.Excel;
using Office = Microsoft.Office.Core;
using Microsoft.Office.Tools.Excel;
using Microsoft.Office.Core;
using ExcelAddIn.Support;
using System.Windows.Forms;
using ExcelAddIn.View;
using ExcelAddIn.ViewModel;

namespace ExcelAddIn
{
    public partial class ThisAddIn
    {       
        private MenuManager _menuManager;
        private VMMainInputControl _vmMainInput;

        private bool CheckLicense()
        {
            return true;
        }

        private void InitializeAddIn()
        {            
            SetupMenu();
            _vmMainInput = new VMMainInputControl();
            _vmMainInput.ExcelApplication = this.Application;            
        }
       
        private void SetupMenu()
        {            
            _menuManager = new MenuManager(this);
            _menuManager.CreateAddInMenu("Dasmic");
            _menuManager.AddMenuItem("Do This", ExecuteMenuItem1Action);
            _menuManager.AddMenuItem("Do That", ExecuteMenuItem2Action);
        }
        
        private MainInput GetMainInputUI()
        {
            MainInput mainInput = new MainInput();
            mainInput.WpfControl.DataContext = _vmMainInput;//Some sheet is loaded by now
            mainInput.TopMost = true;
            this.Application.SheetSelectionChange += new
                       Excel.AppEvents_SheetSelectionChangeEventHandler
                           (_vmMainInput.OnSheetChange);
            return mainInput;
        }

        private void ExecuteMenuItem1Action(Office.CommandBarButton menuItem, ref Boolean CancelDefault)
        {
            MainInput mainInput = GetMainInputUI();           
            mainInput.Show();
          
            string selectedMenu = string.Format("You just selected '{0}' from the menu.", menuItem.Caption);
            string workBookSelected = "No workbook selected";
            if (this.Application.Workbooks.Count > 0)
            {
                workBookSelected = String.Format("The name of your workbook is '{0}'.", this.Application.ActiveWorkbook.Name);
            }
            string message = selectedMenu + "\n" + workBookSelected;
            MessageBox.Show(message, "Add-In Menu Demo", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }

        private void ExecuteMenuItem2Action(Office.CommandBarButton menuItem, ref Boolean CancelDefault)
        {
            string selectedMenu = string.Format("You just selected '{0}' from the menu.", menuItem.Caption);
            string workBookSelected = "No workbook selected";
            if (this.Application.Workbooks.Count > 0)
            {
                workBookSelected = String.Format("The name of your workbook is '{0}'.", this.Application.ActiveWorkbook.Name);
            }
            string message = selectedMenu + "\n" + workBookSelected;
            MessageBox.Show(message, "Add-In Menu Demo", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }


        private void ThisAddIn_Startup(object sender, System.EventArgs e)
        {
            //Check for License          
            InitializeAddIn();
            this.Application.WorkbookBeforeSave += new Microsoft.Office.Interop.Excel.AppEvents_WorkbookBeforeSaveEventHandler(Application_WorkbookBeforeSave);
        }

        private void ThisAddIn_Shutdown(object sender, System.EventArgs e)
        {

        }

        void Application_WorkbookBeforeSave(Microsoft.Office.Interop.Excel.Workbook Wb, bool SaveAsUI, ref bool Cancel)
        {
            Excel.Worksheet activeWorksheet = ((Excel.Worksheet)Application.ActiveSheet);
            Excel.Range firstRow = activeWorksheet.get_Range("A1");
            //firstRow.EntireRow.Insert(Excel.XlInsertShiftDirection.xlShiftDown);
            //Excel.Range newFirstRow = activeWorksheet.get_Range("A1");
            //newFirstRow.Value2 = "This text was added by using code";
        }

        #region VSTO generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InternalStartup()
        {
            this.Startup += new System.EventHandler(ThisAddIn_Startup);
            this.Shutdown += new System.EventHandler(ThisAddIn_Shutdown);
        }
        
        #endregion
    }
}
