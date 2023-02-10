using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace TaskDrawing
{
    public partial class fMain : Form
    {

        private CTaskExecutionTrace _eT;
        private bool _bDrawClicked;
        private int _maxTime;
        private string taskIDPrefix;
        private Font _fTask;
        private Font _fTick;
        private Font _fRelease;

        public fMain()
        {
            InitializeComponent();
            _bDrawClicked = false;
            this.textFileName.Text = @"C:\Users\Chaitanya Belwal\Documents\Research_Repository\Code\TaskGeneration\TaskDrawing\bin\Debug\TaskDrawing.txt";
            
            _fTask = new Font("Arial", 10, FontStyle.Bold);
            _fTick = new Font("Arial", 9, FontStyle.Regular);
            _fRelease = new Font("Arial", 8, FontStyle.Bold);
        }

      

        private void DrawTaskSet(CTaskDrawing td)
        {
            int time=0, timeTick=0;
            CTaskExecutionParam param;
            string taskID;


            
            //Font fBoxText = new Font("Arial",Convert.ToInt32(textTaskFontSize.Text),FontStyle.Bold)

            //Draw Ticks and Tick Text 
            timeTick = Convert.ToInt32(textTimeStart.Text);
            for (time = 0; time <= _maxTime; time++)
            {
                td.DrawTick(time);
                td.AddTickText(time, timeTick++.ToString(),_fTick);
                                
            }
            

            for (time = 0; time <= _eT.maxTime; time++)
            {
                try
                {
                    if (time > _maxTime) break;
                    param = (CTaskExecutionParam)_eT.hshTimeTrace[time];
                    if (param.taskID !=  taskIDPrefix + "E")
                    {
                        taskID = param.taskID;
                        td.DrawBox(time);
                        td.AddBoxText(time, taskID,_fTask);
                        if (param.bEnd) td.DrawCompletionArrow(time + 1);
                    }

                    //Draw Start Arrows

                    if (param.releasingTasks != null)
                    {
                        foreach (string s in param.releasingTasks)
                        {

                            td.DrawReleaseArrow(s, (int)_eT.hshTaskID[s], time);
                            td.AddReleaseArrowText(s, (int)_eT.hshTaskID[s], time, _fRelease);
                        }
                    }
                }
                catch
                {
                    MessageBox.Show("Could not add Task at time :" + time.ToString());
                }
            }
        }

        

        private void btnBrowse_Click(object sender, EventArgs e)
        {
            openFileDialog1.ShowDialog();
            textFileName.Text = openFileDialog1.FileName;
        }

       
        private void pictureBoxMain_Paint(object sender, PaintEventArgs e)
        {
            if (_bDrawClicked) btnDraw_Click(sender, e);
        }


        private void btnDraw_Click(object sender, EventArgs e)
        {
            _bDrawClicked = true;
            Graphics g = pictureBoxMain.CreateGraphics();
            DrawMain(g);  
        }


        private void DrawMain(Graphics g)
        {
            
            CTaskDrawing td = new CTaskDrawing(g);

            taskIDPrefix = textTaskIDPrefix.Text.Trim();
            //Assign Values
            _eT = new CTaskExecutionTrace();
            _eT.ReadTaskExecutionTraceFile(textFileName.Text,taskIDPrefix);

            //Set Basic Parameters

            td.height = this.pictureBoxMain.Height;
            td.width = this.pictureBoxMain.Width;
            td._tickWidth = Convert.ToInt32(textTickWidth.Text);
            td._boxHeight = Convert.ToInt32(textBoxHeight.Text);
            td._XaxisEdgeBuffer = Convert.ToInt32(textXAxisEdgeBuffer.Text);
            td._YaxisEdgeBuffer = Convert.ToInt32(textYAxisEdgeBuffer.Text);
            td._releaseArrowSpacing = Convert.ToInt32(textReleaseArrowSpacing.Text);
            
            td._tickHeight = Convert.ToInt32(textTickMarkerHeight.Text);
            _maxTime = Convert.ToInt32(textMaximumTicks.Text);

            //Test();  
           

            td.DrawAxis();
            DrawTaskSet(td);
            
        }


        private void btnCopyToClipboard_Click(object sender, EventArgs e)
        {
            Bitmap bmp = new Bitmap(pictureBoxMain.Size.Width, pictureBoxMain.Size.Height);
            Graphics g = Graphics.FromImage(bmp);
            DrawMain(g); 
            Clipboard.SetData(DataFormats.Bitmap, bmp);
        }

      
    

        private void btnTaskFont_Click(object sender, EventArgs e)
        {
            fontDialog1.Font = _fTask;
            fontDialog1.ShowDialog();
            _fTask = fontDialog1.Font;
        }

        private void btnTickFont_Click(object sender, EventArgs e)
        {
            fontDialog1.Font = _fTick;
            fontDialog1.ShowDialog();
            _fTick = fontDialog1.Font;
        }

        private void btnReleaseFont_Click(object sender, EventArgs e)
        {
            fontDialog1.Font = _fRelease;
            fontDialog1.ShowDialog();
            _fRelease = fontDialog1.Font;
        }

        private void fMain_Load(object sender, EventArgs e)
        {

        }

        

        

        


    }
}
