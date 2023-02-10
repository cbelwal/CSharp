namespace TestLibUI
{
    partial class fBoundedPeriods
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.groupBox1 = new System.Windows.Forms.GroupBox();
            this.comboPeriodMethod = new System.Windows.Forms.ComboBox();
            this.label9 = new System.Windows.Forms.Label();
            this.label8 = new System.Windows.Forms.Label();
            this.textTasksToGen = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.textMaximumTaskPeriod = new System.Windows.Forms.TextBox();
            this.textMinTaskPeriod = new System.Windows.Forms.TextBox();
            this.textTasksPerSet = new System.Windows.Forms.TextBox();
            this.btnStart = new System.Windows.Forms.Button();
            this.label4 = new System.Windows.Forms.Label();
            this.label3 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.btnClipboard = new System.Windows.Forms.Button();
            this.listMain = new System.Windows.Forms.ListBox();
            this.label6 = new System.Windows.Forms.Label();
            this.textNumberOfPossibleTaskSets = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.textLCM = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.textNumberOfTaskSets = new System.Windows.Forms.TextBox();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel3 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLabel2 = new System.Windows.Forms.ToolStripStatusLabel();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.comboPeriodMethod);
            this.groupBox1.Controls.Add(this.label9);
            this.groupBox1.Controls.Add(this.label8);
            this.groupBox1.Controls.Add(this.textTasksToGen);
            this.groupBox1.Controls.Add(this.label7);
            this.groupBox1.Controls.Add(this.textMaximumTaskPeriod);
            this.groupBox1.Controls.Add(this.textMinTaskPeriod);
            this.groupBox1.Controls.Add(this.textTasksPerSet);
            this.groupBox1.Controls.Add(this.btnStart);
            this.groupBox1.Controls.Add(this.label4);
            this.groupBox1.Controls.Add(this.label3);
            this.groupBox1.Controls.Add(this.label2);
            this.groupBox1.Location = new System.Drawing.Point(6, 3);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(332, 192);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Task set parameters";
            // 
            // comboPeriodMethod
            // 
            this.comboPeriodMethod.FormattingEnabled = true;
            this.comboPeriodMethod.Items.AddRange(new object[] {
            "Combinatorial",
            "Random",
            "All Possible"});
            this.comboPeriodMethod.Location = new System.Drawing.Point(180, 139);
            this.comboPeriodMethod.Name = "comboPeriodMethod";
            this.comboPeriodMethod.Size = new System.Drawing.Size(131, 21);
            this.comboPeriodMethod.TabIndex = 19;
            // 
            // label9
            // 
            this.label9.AutoSize = true;
            this.label9.Location = new System.Drawing.Point(8, 144);
            this.label9.Name = "label9";
            this.label9.Size = new System.Drawing.Size(131, 13);
            this.label9.TabIndex = 18;
            this.label9.Text = "Period Generation Method";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(8, 60);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(171, 13);
            this.label8.TabIndex = 17;
            this.label8.Text = "(Valid only for Random Generation)";
            // 
            // textTasksToGen
            // 
            this.textTasksToGen.Location = new System.Drawing.Point(179, 16);
            this.textTasksToGen.MaxLength = 6;
            this.textTasksToGen.Name = "textTasksToGen";
            this.textTasksToGen.Size = new System.Drawing.Size(133, 20);
            this.textTasksToGen.TabIndex = 16;
            this.textTasksToGen.Text = "100";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(8, 16);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(124, 13);
            this.label7.TabIndex = 15;
            this.label7.Text = "# Task Sets to Generate";
            // 
            // textMaximumTaskPeriod
            // 
            this.textMaximumTaskPeriod.Location = new System.Drawing.Point(179, 113);
            this.textMaximumTaskPeriod.MaxLength = 6;
            this.textMaximumTaskPeriod.Name = "textMaximumTaskPeriod";
            this.textMaximumTaskPeriod.Size = new System.Drawing.Size(133, 20);
            this.textMaximumTaskPeriod.TabIndex = 14;
            this.textMaximumTaskPeriod.Text = "80";
            // 
            // textMinTaskPeriod
            // 
            this.textMinTaskPeriod.Location = new System.Drawing.Point(179, 89);
            this.textMinTaskPeriod.MaxLength = 6;
            this.textMinTaskPeriod.Name = "textMinTaskPeriod";
            this.textMinTaskPeriod.Size = new System.Drawing.Size(133, 20);
            this.textMinTaskPeriod.TabIndex = 13;
            this.textMinTaskPeriod.Text = "50";
            // 
            // textTasksPerSet
            // 
            this.textTasksPerSet.Location = new System.Drawing.Point(179, 39);
            this.textTasksPerSet.MaxLength = 3;
            this.textTasksPerSet.Name = "textTasksPerSet";
            this.textTasksPerSet.Size = new System.Drawing.Size(133, 20);
            this.textTasksPerSet.TabIndex = 12;
            this.textTasksPerSet.Text = "3";
            // 
            // btnStart
            // 
            this.btnStart.Location = new System.Drawing.Point(214, 164);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(98, 22);
            this.btnStart.TabIndex = 10;
            this.btnStart.Text = "&Start";
            this.btnStart.UseVisualStyleBackColor = true;
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(8, 39);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(104, 13);
            this.label4.TabIndex = 6;
            this.label4.Text = "# Tasks in Each Set";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(8, 120);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(111, 13);
            this.label3.TabIndex = 4;
            this.label3.Text = "Maximum Task Period";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(8, 96);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(108, 13);
            this.label2.TabIndex = 2;
            this.label2.Text = "Minimum Task Period";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.btnClipboard);
            this.groupBox2.Controls.Add(this.listMain);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Controls.Add(this.textNumberOfPossibleTaskSets);
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.textLCM);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Controls.Add(this.textNumberOfTaskSets);
            this.groupBox2.Location = new System.Drawing.Point(6, 201);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(330, 380);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Results";
            // 
            // btnClipboard
            // 
            this.btnClipboard.Location = new System.Drawing.Point(210, 351);
            this.btnClipboard.Name = "btnClipboard";
            this.btnClipboard.Size = new System.Drawing.Size(114, 27);
            this.btnClipboard.TabIndex = 11;
            this.btnClipboard.Text = "Copy To Clipboard";
            this.btnClipboard.UseVisualStyleBackColor = true;
            this.btnClipboard.Click += new System.EventHandler(this.btnClipboard_Click);
            // 
            // listMain
            // 
            this.listMain.FormattingEnabled = true;
            this.listMain.Location = new System.Drawing.Point(6, 94);
            this.listMain.Name = "listMain";
            this.listMain.Size = new System.Drawing.Size(318, 251);
            this.listMain.TabIndex = 10;
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(3, 48);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(195, 13);
            this.label6.TabIndex = 9;
            this.label6.Text = "Number of Period Sets Possible (Bound)";
            // 
            // textNumberOfPossibleTaskSets
            // 
            this.textNumberOfPossibleTaskSets.Enabled = false;
            this.textNumberOfPossibleTaskSets.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textNumberOfPossibleTaskSets.Location = new System.Drawing.Point(223, 44);
            this.textNumberOfPossibleTaskSets.Name = "textNumberOfPossibleTaskSets";
            this.textNumberOfPossibleTaskSets.Size = new System.Drawing.Size(101, 20);
            this.textNumberOfPossibleTaskSets.TabIndex = 8;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(3, 71);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(76, 13);
            this.label1.TabIndex = 7;
            this.label1.Text = "Maximum LCM";
            // 
            // textLCM
            // 
            this.textLCM.Enabled = false;
            this.textLCM.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textLCM.Location = new System.Drawing.Point(223, 68);
            this.textLCM.Name = "textLCM";
            this.textLCM.Size = new System.Drawing.Size(101, 20);
            this.textLCM.TabIndex = 6;
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(3, 21);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(205, 13);
            this.label5.TabIndex = 5;
            this.label5.Text = "Number of Period Sets Generated (Actual)";
            // 
            // textNumberOfTaskSets
            // 
            this.textNumberOfTaskSets.Enabled = false;
            this.textNumberOfTaskSets.Font = new System.Drawing.Font("Microsoft Sans Serif", 8.25F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.textNumberOfTaskSets.Location = new System.Drawing.Point(223, 18);
            this.textNumberOfTaskSets.Name = "textNumberOfTaskSets";
            this.textNumberOfTaskSets.Size = new System.Drawing.Size(101, 20);
            this.textNumberOfTaskSets.TabIndex = 1;
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripStatusLabel3,
            this.toolStripStatusLabel2});
            this.statusStrip1.Location = new System.Drawing.Point(0, 584);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(344, 22);
            this.statusStrip1.TabIndex = 2;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(38, 17);
            this.toolStripStatusLabel1.Text = "Ready";
            // 
            // toolStripStatusLabel3
            // 
            this.toolStripStatusLabel3.Name = "toolStripStatusLabel3";
            this.toolStripStatusLabel3.Size = new System.Drawing.Size(13, 17);
            this.toolStripStatusLabel3.Text = "  ";
            // 
            // toolStripStatusLabel2
            // 
            this.toolStripStatusLabel2.Name = "toolStripStatusLabel2";
            this.toolStripStatusLabel2.Size = new System.Drawing.Size(157, 17);
            this.toolStripStatusLabel2.Text = "Comments: cbelwal@cs.uh.edu";
            this.toolStripStatusLabel2.TextAlign = System.Drawing.ContentAlignment.MiddleRight;
            // 
            // fBoundedPeriods
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(344, 606);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedSingle;
            this.MaximizeBox = false;
            this.Name = "fBoundedPeriods";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Period Generation (PG)-Tool";
            this.Load += new System.EventHandler(this.fBoundedPeriods_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.Button btnStart;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox textMaximumTaskPeriod;
        private System.Windows.Forms.TextBox textMinTaskPeriod;
        private System.Windows.Forms.TextBox textTasksPerSet;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.TextBox textNumberOfTaskSets;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel2;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel3;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.TextBox textLCM;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox textNumberOfPossibleTaskSets;
        private System.Windows.Forms.ListBox listMain;
        private System.Windows.Forms.Button btnClipboard;
        private System.Windows.Forms.Label label9;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.TextBox textTasksToGen;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.ComboBox comboPeriodMethod;
    }
}

