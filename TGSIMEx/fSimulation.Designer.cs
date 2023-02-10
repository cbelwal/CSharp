namespace TaskGen
{
    partial class fSimulation
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
            this.btnCopyToClipboard_1 = new System.Windows.Forms.Button();
            this.textTaskSetSchedulability = new System.Windows.Forms.TextBox();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.label1 = new System.Windows.Forms.Label();
            this.btnBrowseTraceFolder = new System.Windows.Forms.Button();
            this.textTraceFolder = new System.Windows.Forms.TextBox();
            this.checkBoxDisplayDiscreteTimeExecution = new System.Windows.Forms.CheckBox();
            this.btnCopyToClipboard_2 = new System.Windows.Forms.Button();
            this.textDiscreteTaskExecution = new System.Windows.Forms.TextBox();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.btnBrowse = new System.Windows.Forms.Button();
            this.textFolderName = new System.Windows.Forms.TextBox();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.btnStart = new System.Windows.Forms.Button();
            this.btnStop = new System.Windows.Forms.Button();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.exitToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.groupBox1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnCopyToClipboard_1);
            this.groupBox1.Controls.Add(this.textTaskSetSchedulability);
            this.groupBox1.Location = new System.Drawing.Point(7, 84);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(709, 136);
            this.groupBox1.TabIndex = 0;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Task Set Schedulability";
            // 
            // btnCopyToClipboard_1
            // 
            this.btnCopyToClipboard_1.Location = new System.Drawing.Point(605, 98);
            this.btnCopyToClipboard_1.Name = "btnCopyToClipboard_1";
            this.btnCopyToClipboard_1.Size = new System.Drawing.Size(98, 29);
            this.btnCopyToClipboard_1.TabIndex = 1;
            this.btnCopyToClipboard_1.Text = "Copy to &Clipboard";
            this.btnCopyToClipboard_1.UseVisualStyleBackColor = true;
            this.btnCopyToClipboard_1.Click += new System.EventHandler(this.btnCopyToClipboard_1_Click);
            // 
            // textTaskSetSchedulability
            // 
            this.textTaskSetSchedulability.Location = new System.Drawing.Point(5, 19);
            this.textTaskSetSchedulability.Multiline = true;
            this.textTaskSetSchedulability.Name = "textTaskSetSchedulability";
            this.textTaskSetSchedulability.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textTaskSetSchedulability.Size = new System.Drawing.Size(594, 108);
            this.textTaskSetSchedulability.TabIndex = 0;
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.label1);
            this.groupBox2.Controls.Add(this.btnBrowseTraceFolder);
            this.groupBox2.Controls.Add(this.textTraceFolder);
            this.groupBox2.Controls.Add(this.checkBoxDisplayDiscreteTimeExecution);
            this.groupBox2.Controls.Add(this.btnCopyToClipboard_2);
            this.groupBox2.Controls.Add(this.textDiscreteTaskExecution);
            this.groupBox2.Location = new System.Drawing.Point(7, 226);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(709, 315);
            this.groupBox2.TabIndex = 1;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Task Execution at Discrete Times";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(5, 42);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(391, 13);
            this.label1.TabIndex = 12;
            this.label1.Text = "Please select a valid folder name or leave blank if no trace files have to be wri" +
                "tten";
            // 
            // btnBrowseTraceFolder
            // 
            this.btnBrowseTraceFolder.Location = new System.Drawing.Point(625, 14);
            this.btnBrowseTraceFolder.Name = "btnBrowseTraceFolder";
            this.btnBrowseTraceFolder.Size = new System.Drawing.Size(78, 25);
            this.btnBrowseTraceFolder.TabIndex = 11;
            this.btnBrowseTraceFolder.Text = "&Browse";
            this.btnBrowseTraceFolder.UseVisualStyleBackColor = true;
            this.btnBrowseTraceFolder.Click += new System.EventHandler(this.btnBrowseTraceFolder_Click);
            // 
            // textTraceFolder
            // 
            this.textTraceFolder.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textTraceFolder.Location = new System.Drawing.Point(6, 19);
            this.textTraceFolder.Name = "textTraceFolder";
            this.textTraceFolder.Size = new System.Drawing.Size(613, 20);
            this.textTraceFolder.TabIndex = 10;
            // 
            // checkBoxDisplayDiscreteTimeExecution
            // 
            this.checkBoxDisplayDiscreteTimeExecution.AutoSize = true;
            this.checkBoxDisplayDiscreteTimeExecution.Location = new System.Drawing.Point(605, 257);
            this.checkBoxDisplayDiscreteTimeExecution.Name = "checkBoxDisplayDiscreteTimeExecution";
            this.checkBoxDisplayDiscreteTimeExecution.Size = new System.Drawing.Size(60, 17);
            this.checkBoxDisplayDiscreteTimeExecution.TabIndex = 3;
            this.checkBoxDisplayDiscreteTimeExecution.Text = "Display";
            this.checkBoxDisplayDiscreteTimeExecution.UseVisualStyleBackColor = true;
            // 
            // btnCopyToClipboard_2
            // 
            this.btnCopyToClipboard_2.Location = new System.Drawing.Point(605, 280);
            this.btnCopyToClipboard_2.Name = "btnCopyToClipboard_2";
            this.btnCopyToClipboard_2.Size = new System.Drawing.Size(98, 29);
            this.btnCopyToClipboard_2.TabIndex = 2;
            this.btnCopyToClipboard_2.Text = "Copy to &Clipboard";
            this.btnCopyToClipboard_2.UseVisualStyleBackColor = true;
            this.btnCopyToClipboard_2.Click += new System.EventHandler(this.btnCopyToClipboard_2_Click);
            // 
            // textDiscreteTaskExecution
            // 
            this.textDiscreteTaskExecution.Location = new System.Drawing.Point(8, 68);
            this.textDiscreteTaskExecution.Multiline = true;
            this.textDiscreteTaskExecution.Name = "textDiscreteTaskExecution";
            this.textDiscreteTaskExecution.ScrollBars = System.Windows.Forms.ScrollBars.Vertical;
            this.textDiscreteTaskExecution.Size = new System.Drawing.Size(591, 241);
            this.textDiscreteTaskExecution.TabIndex = 0;
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.btnBrowse);
            this.groupBox3.Controls.Add(this.textFolderName);
            this.groupBox3.Location = new System.Drawing.Point(7, 27);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(709, 51);
            this.groupBox3.TabIndex = 42;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Task Set Folder";
            // 
            // btnBrowse
            // 
            this.btnBrowse.Location = new System.Drawing.Point(625, 14);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(78, 25);
            this.btnBrowse.TabIndex = 9;
            this.btnBrowse.Text = "&Browse";
            this.btnBrowse.UseVisualStyleBackColor = true;
            this.btnBrowse.Click += new System.EventHandler(this.btnBrowse_Click);
            // 
            // textFolderName
            // 
            this.textFolderName.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textFolderName.Location = new System.Drawing.Point(6, 19);
            this.textFolderName.Name = "textFolderName";
            this.textFolderName.Size = new System.Drawing.Size(613, 20);
            this.textFolderName.TabIndex = 8;
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1});
            this.statusStrip1.Location = new System.Drawing.Point(0, 576);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(721, 22);
            this.statusStrip1.TabIndex = 43;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(109, 17);
            this.toolStripStatusLabel1.Text = "toolStripStatusLabel1";
            // 
            // btnStart
            // 
            this.btnStart.Location = new System.Drawing.Point(628, 547);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(88, 26);
            this.btnStart.TabIndex = 44;
            this.btnStart.Text = "&Start";
            this.btnStart.UseVisualStyleBackColor = true;
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // btnStop
            // 
            this.btnStop.Location = new System.Drawing.Point(534, 547);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(88, 26);
            this.btnStop.TabIndex = 45;
            this.btnStop.Text = "&Stop";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(721, 24);
            this.menuStrip1.TabIndex = 46;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.exitToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(35, 20);
            this.fileToolStripMenuItem.Text = "&File";
            // 
            // exitToolStripMenuItem
            // 
            this.exitToolStripMenuItem.Name = "exitToolStripMenuItem";
            this.exitToolStripMenuItem.Size = new System.Drawing.Size(103, 22);
            this.exitToolStripMenuItem.Text = "&Exit";
            this.exitToolStripMenuItem.Click += new System.EventHandler(this.exitToolStripMenuItem_Click);
            // 
            // fSimulation
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(721, 598);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.btnStart);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.menuStrip1);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.FixedToolWindow;
            this.MainMenuStrip = this.menuStrip1;
            this.Name = "fSimulation";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "fSimulation";
            this.Load += new System.EventHandler(this.fSimulation_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox3.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.TextBox textTaskSetSchedulability;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox textDiscreteTaskExecution;
        private System.Windows.Forms.Button btnCopyToClipboard_1;
        private System.Windows.Forms.Button btnCopyToClipboard_2;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.TextBox textFolderName;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.Button btnStart;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem exitToolStripMenuItem;
        private System.Windows.Forms.CheckBox checkBoxDisplayDiscreteTimeExecution;
        private System.Windows.Forms.Button btnBrowseTraceFolder;
        private System.Windows.Forms.TextBox textTraceFolder;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
    }
}