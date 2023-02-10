using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Drawing;

namespace TaskDrawing
{
    public class CTaskDrawing
    {
        public int width;
        public int height;
        public int _XaxisEdgeBuffer;
        public int _YaxisEdgeBuffer;
        public int _tickWidth;
        public int _boxHeight;
        public int _releaseArrowSpacing;
        public int _originX;
        public int _originY;
        
        public int _tickHeight;
        
        Graphics _gMain;
        public CTaskDrawing(Graphics g)
        {
            _gMain = g;
            _gMain.Clear(Color.White);
        }
        
        
        public void DrawAxis()
        {
            Pen p = new Pen(Color.Black, 1);

            //Y-Axis
            _originX = _YaxisEdgeBuffer;// _XaxisEdgeBuffer;         
            //X Axis
            _originY = height - (_XaxisEdgeBuffer);
           
        }

        public void DrawTick(int iCount)
        {
            int x, y_up, y_down;
            Pen p = new Pen(Color.Black, 1);

            x = iCount * _tickWidth + _originX;
            y_up = _originY - (_tickHeight / 2);
            y_down = _originY + (_tickHeight / 2);
            _gMain.DrawLine(p, x, y_up, x, y_down);

            //Draw Horizontal Line
            _gMain.DrawLine(p, x, _originY, x + _tickWidth, _originY);
        }

        public void AddTickText(int iCount, string tickText,Font font)
        {
            int x;
            //Font f = null;

            //if (FontType.ToUpper() == "BOLD")
            //    f = new Font("Arial", Size, FontStyle.Bold);

            //if (FontType.ToUpper() == "REGULAR")
            //    f = new Font("Arial", Size, FontStyle.Regular);

            Pen p = new Pen(Color.Black, 1);

            x = iCount * _tickWidth + _originX;
            _gMain.DrawString(tickText, font, Brushes.Black, x - 5, _originY + 7);
        }

        public void AddBoxText(int iCount, string Text, Font font)
        {
            int x;
            //Font f = font;//new Font("Arial", 10, FontStyle.Bold);
            Pen p = new Pen(Color.Black, 1);

            x = iCount * _tickWidth + _originX;
            _gMain.DrawString(Text, font, Brushes.Black, x + (_tickWidth / 2) - 10, _originY - (_boxHeight / 2) - 10);
        }



        public void DrawBox(int iCount)
        {
            int x;
            Pen p = new Pen(Color.Black, 1);

            x = iCount * _tickWidth + _originX;

            _gMain.DrawRectangle(p, x, _originY - _boxHeight, _tickWidth, _boxHeight);

        }

        public void DrawReleaseArrow(string taskID, int taskIdx, int iCount)
        {
            int x;
            
            int arrowTop;
            Pen p = new Pen(Color.Black, 1);

            //taskIdx range from 0 onwards

            x = iCount * _tickWidth + _originX;

            //lineTop = _originY - _boxHeight * 2;
            arrowTop = _originY - _boxHeight - _releaseArrowSpacing
                    - taskIdx * _releaseArrowSpacing; //Release Arrow Spacing

            //Draw Straight Line
            _gMain.DrawLine(p, x, arrowTop, x, _originY);

            //Draw Left Slant
            _gMain.DrawLine(p, x, arrowTop, x - 5, arrowTop + 5);
            //Draw Right Slant
            _gMain.DrawLine(p, x, arrowTop, x + 5, arrowTop + 5);
        }


        //Task ID will automatically be accessible
        public void AddReleaseArrowText(string taskID,int taskIdx, int iCount, Font font)
        {
            int x;
            int arrowTop, textTop;
            Pen p = new Pen(Color.Black, 1);
            
            //Font f = new Font("Arial", 8, FontStyle.Bold);

            x = iCount * _tickWidth + _originX - 1;

            arrowTop = _originY - _boxHeight - _releaseArrowSpacing
                    - taskIdx * _releaseArrowSpacing; //Release Arrow Spacing
            textTop = arrowTop - 3;
            x = x + 5;
            _gMain.DrawString(taskID, font, Brushes.Black, x, textTop);
        }


        //Will be Drawn in Center
        public void DrawCompletionArrow(int iCount)
        {
            int x;
            int arrowTop;
            Pen p = new Pen(Color.Black, 1);

            x = iCount * _tickWidth + _originX;
            arrowTop = _originY - (_boxHeight / 2) + 12; //Slightly below center

            //Draw Straight Line
            _gMain.DrawLine(p, x, _originY - _boxHeight, x, _originY);


            //Draw Left Slant
            _gMain.DrawLine(p, x, arrowTop, x - 5, arrowTop - 5);

            //Draw Right Slant
            _gMain.DrawLine(p, x, arrowTop, x + 5, arrowTop - 5);
        }

        public void Test()
        {

            DrawTick(1);
            DrawTick(2);
            DrawBox(1);
            DrawBox(2);
            //DrawReleaseArrow(1);
            //DrawReleaseArrow(4);
            DrawCompletionArrow(4);
            DrawCompletionArrow(3);
            //AddTickText(1, "10");
            //AddBoxText(1, "T1");
            //AddBoxText(2, "T2");
            //Clipboard.SetData(DataFormats.Bitmap,pictureBoxMain.DrawToBitmap)
        }
    }
}
