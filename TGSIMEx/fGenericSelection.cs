using System;
using System.Collections.Generic;
using System.Collections;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;
using TGSIMExLib;

namespace TaskGen
{
    partial class fGenericSelection : Form
    {
      

        
        public fGenericSelection()
        {
            InitializeComponent();
        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="sectionHeading"></param>
        /// 
        /// <param name="interfaceObj"></param>
        public List<CListBoxWrapper> DisplayForm(string sectionHeading, Type typeObj, 
                                        CListBoxWrapper currentSelection, bool bHide)
        {
            ArrayList allObjs;
            CObjectFactory factory = new CObjectFactory();
            CListBoxWrapper oWrap;
            List<CListBoxWrapper> allSelected = new List<CListBoxWrapper>();

            this.Text = "Make Selection";

            groupBoxMain.Text = sectionHeading;

            allObjs = factory.GetObjects(typeObj, Application.StartupPath);

            if (allObjs == null) return allSelected; //Count will be 0

            foreach (Object o in allObjs)
            {
                oWrap = new CListBoxWrapper();
                oWrap.objMain = (IRootInterface)o;
                oWrap.index = listBoxMain.Items.Count;
                listBoxMain.Items.Add(oWrap);  
            }

            //Set Current Selection
            if (currentSelection != null)
            {
                if (currentSelection.index < listBoxMain.Items.Count)
                {
                    listBoxMain.SelectedIndex = currentSelection.index;
                    listBoxMain.SetItemChecked(listBoxMain.SelectedIndex, true);
                }
            }

            if(!bHide) this.ShowDialog();

            //Return all Selected
            
            foreach (CListBoxWrapper tmpO in listBoxMain.CheckedItems)
                allSelected.Add(tmpO);

            return allSelected;

        }

        private void btnOk_Click(object sender, EventArgs e)
        {
            this.Close();
        }

      
    }
}
