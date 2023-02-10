using System;
using System.Collections.Generic;
using System.Windows.Forms;
using Office = Microsoft.Office.Core;

namespace ExcelAddIn.Support
{
    internal class MenuManager
    {
        #region Members
        public delegate void OnMenuItemClick(Office.CommandBarButton menuItem, ref Boolean CancelDefault);

        private ThisAddIn m_AddIn;
        private Office.CommandBar m_ExcelMenuBar;
        private Office.CommandBarControl m_AddInMenu;
        private List<Office.CommandBarButton> m_MenuItems;

        #endregion

        #region Constructor

        internal MenuManager(ThisAddIn addIn)
        {
            m_AddIn = addIn;
            m_MenuItems = new List<Microsoft.Office.Core.CommandBarButton>();
            InitializeExcelMenuBar();
        }

        #endregion

        internal void CreateAddInMenu(string menuCaption)
        {
            try
            {
                Office.CommandBarControl menu = m_ExcelMenuBar.Controls.Add(
                    Office.MsoControlType.msoControlPopup,
                    Type.Missing, Type.Missing, Type.Missing, true);
                menu.Caption = menuCaption;
                m_AddInMenu = menu;
               
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, ex.Source, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        internal void AddMenuItem(string menuItemCaption, OnMenuItemClick onMenuItemClicked)
        {
            Office.CommandBarButton menuItem = CreateMenuItem((Office.CommandBarPopup)m_AddInMenu, menuItemCaption);
            m_MenuItems.Add(menuItem);
            SubscribeMenuItemClick(menuItem,onMenuItemClicked);
        }

        #region helper methods

        private void InitializeExcelMenuBar()
        {
            try
            {
                m_ExcelMenuBar = m_AddIn.Application.CommandBars["Worksheet Menu Bar"];
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, ex.Source, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private Office.CommandBarButton CreateMenuItem(Office.CommandBarPopup parentMenu, string menuItemCaption)
        {
            Office.CommandBarControl cbc = null;
            try
            {
                cbc = parentMenu.Controls.Add(
                    Office.MsoControlType.msoControlButton, Type.Missing,
                    Type.Missing, Type.Missing, true);
                cbc.Caption = menuItemCaption;
                cbc.Visible = true;

            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message,
                    ex.Source, MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            return (Office.CommandBarButton)cbc;
        }

        #endregion

        #region handling of events

        //private void MenuItem_Click(Office.CommandBarButton menuItem, ref Boolean CancelDefault)
        //{
        //    ExecuteMenuItemAction(menuItem);
        //}

        /*private void ExecuteMenuItemAction(Office.CommandBarButton menuItem)
        {
            string selectedMenu = string.Format("You just selected '{0}' from the menu.", menuItem.Caption);
            string workBookSelected = "No workbook selected";
            if (m_AddIn.Application.Workbooks.Count > 0)
            {
                workBookSelected = String.Format("The name of your workbook is '{0}'.", m_AddIn.Application.ActiveWorkbook.Name);
            }
            string message = selectedMenu + "\n" + workBookSelected;
            MessageBox.Show(message, "Add-In Menu Demo", MessageBoxButtons.OK, MessageBoxIcon.Information);
        }*/

        private void SubscribeMenuItemClick(Office.CommandBarButton menuItem, OnMenuItemClick onMenuItemClicked)
        {
            menuItem.Click += new Microsoft.Office.Core._CommandBarButtonEvents_ClickEventHandler(onMenuItemClicked);
        }

        private void UnsubscribeMenuItemClick(Office.CommandBarButton menuItem, OnMenuItemClick onMenuItemClicked)
        {
            //menuItem.Click -= new Microsoft.Office.Core._CommandBarButtonEvents_ClickEventHandler();
        }

        internal void UnsubscribeAll()
        {
            foreach (Office.CommandBarButton menuItem in m_MenuItems)
            {
                //UnsubscribeMenuItemClick(menuItem);
            }
        }

        #endregion
    }
}



