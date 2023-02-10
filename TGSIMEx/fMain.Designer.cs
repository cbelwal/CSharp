namespace TaskGen
{
    partial class fMain
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
            this.btnBrowse = new System.Windows.Forms.Button();
            this.textFolderName = new System.Windows.Forms.TextBox();
            this.statusStrip1 = new System.Windows.Forms.StatusStrip();
            this.toolStripStatusLabel1 = new System.Windows.Forms.ToolStripStatusLabel();
            this.toolStripStatusLeft = new System.Windows.Forms.ToolStripStatusLabel();
            this.groupBox2 = new System.Windows.Forms.GroupBox();
            this.checkBoxPruneFolder = new System.Windows.Forms.CheckBox();
            this.checkUniquePeriod = new System.Windows.Forms.CheckBox();
            this.checkUniqueExecutionTime = new System.Windows.Forms.CheckBox();
            this.textMinimumOffset = new System.Windows.Forms.TextBox();
            this.label7 = new System.Windows.Forms.Label();
            this.textMaximumOffset = new System.Windows.Forms.TextBox();
            this.label8 = new System.Windows.Forms.Label();
            this.checkRunSimulation = new System.Windows.Forms.CheckBox();
            this.textMinimumPeriod = new System.Windows.Forms.TextBox();
            this.label4 = new System.Windows.Forms.Label();
            this.textMinimumExecutionTime = new System.Windows.Forms.TextBox();
            this.label2 = new System.Windows.Forms.Label();
            this.textNoOfTasks = new System.Windows.Forms.TextBox();
            this.label10 = new System.Windows.Forms.Label();
            this.textMaximumExecutionTime = new System.Windows.Forms.TextBox();
            this.label6 = new System.Windows.Forms.Label();
            this.textMaximumPeriod = new System.Windows.Forms.TextBox();
            this.label3 = new System.Windows.Forms.Label();
            this.textNoOfTaskSets = new System.Windows.Forms.TextBox();
            this.label5 = new System.Windows.Forms.Label();
            this.folderBrowserDialog1 = new System.Windows.Forms.FolderBrowserDialog();
            this.menuStrip1 = new System.Windows.Forms.MenuStrip();
            this.fileToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.loadConfigurationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.saveConfigurationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.optionsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.setPeriodGeneratorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.setExecutionTimeGeneratorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.setRelToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.setTaskSetGeneratorToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolStripSeparator1 = new System.Windows.Forms.ToolStripSeparator();
            this.setExecutionModelToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.setExecutionTraceWriterToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.toolsToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.simulationToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.aboutToolStripMenuItem = new System.Windows.Forms.ToolStripMenuItem();
            this.btnStop = new System.Windows.Forms.Button();
            this.btnStart = new System.Windows.Forms.Button();
            this.groupBox3 = new System.Windows.Forms.GroupBox();
            this.checkedListBoxSchedulabilityTests = new System.Windows.Forms.CheckedListBox();
            this.groupBox4 = new System.Windows.Forms.GroupBox();
            this.checkedListBoxSatisfiabilityConditions = new System.Windows.Forms.CheckedListBox();
            this.groupBox5 = new System.Windows.Forms.GroupBox();
            this.checkedListBoxFileFormatter = new System.Windows.Forms.CheckedListBox();
            this.groupBox6 = new System.Windows.Forms.GroupBox();
            this.textBoxMessages = new System.Windows.Forms.TextBox();
            this.label1 = new System.Windows.Forms.Label();
            this.groupBox1.SuspendLayout();
            this.statusStrip1.SuspendLayout();
            this.groupBox2.SuspendLayout();
            this.menuStrip1.SuspendLayout();
            this.groupBox3.SuspendLayout();
            this.groupBox4.SuspendLayout();
            this.groupBox5.SuspendLayout();
            this.groupBox6.SuspendLayout();
            this.SuspendLayout();
            // 
            // groupBox1
            // 
            this.groupBox1.Controls.Add(this.btnBrowse);
            this.groupBox1.Controls.Add(this.textFolderName);
            this.groupBox1.Location = new System.Drawing.Point(2, 27);
            this.groupBox1.Name = "groupBox1";
            this.groupBox1.Size = new System.Drawing.Size(748, 69);
            this.groupBox1.TabIndex = 41;
            this.groupBox1.TabStop = false;
            this.groupBox1.Text = "Folder to store schedulable  task sets";
            // 
            // btnBrowse
            // 
            this.btnBrowse.Location = new System.Drawing.Point(665, 42);
            this.btnBrowse.Name = "btnBrowse";
            this.btnBrowse.Size = new System.Drawing.Size(78, 21);
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
            this.textFolderName.Size = new System.Drawing.Size(737, 20);
            this.textFolderName.TabIndex = 8;
            // 
            // statusStrip1
            // 
            this.statusStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.toolStripStatusLabel1,
            this.toolStripStatusLeft});
            this.statusStrip1.Location = new System.Drawing.Point(0, 662);
            this.statusStrip1.Name = "statusStrip1";
            this.statusStrip1.Size = new System.Drawing.Size(758, 22);
            this.statusStrip1.TabIndex = 43;
            this.statusStrip1.Text = "statusStrip1";
            // 
            // toolStripStatusLabel1
            // 
            this.toolStripStatusLabel1.Name = "toolStripStatusLabel1";
            this.toolStripStatusLabel1.Size = new System.Drawing.Size(0, 17);
            // 
            // toolStripStatusLeft
            // 
            this.toolStripStatusLeft.Name = "toolStripStatusLeft";
            this.toolStripStatusLeft.Size = new System.Drawing.Size(118, 17);
            this.toolStripStatusLeft.Text = "toolStripStatusLabel2";
            // 
            // groupBox2
            // 
            this.groupBox2.Controls.Add(this.checkBoxPruneFolder);
            this.groupBox2.Controls.Add(this.checkUniquePeriod);
            this.groupBox2.Controls.Add(this.checkUniqueExecutionTime);
            this.groupBox2.Controls.Add(this.textMinimumOffset);
            this.groupBox2.Controls.Add(this.label7);
            this.groupBox2.Controls.Add(this.textMaximumOffset);
            this.groupBox2.Controls.Add(this.label8);
            this.groupBox2.Controls.Add(this.checkRunSimulation);
            this.groupBox2.Controls.Add(this.textMinimumPeriod);
            this.groupBox2.Controls.Add(this.label4);
            this.groupBox2.Controls.Add(this.textMinimumExecutionTime);
            this.groupBox2.Controls.Add(this.label2);
            this.groupBox2.Controls.Add(this.textNoOfTasks);
            this.groupBox2.Controls.Add(this.label10);
            this.groupBox2.Controls.Add(this.textMaximumExecutionTime);
            this.groupBox2.Controls.Add(this.label6);
            this.groupBox2.Controls.Add(this.textMaximumPeriod);
            this.groupBox2.Controls.Add(this.label3);
            this.groupBox2.Controls.Add(this.textNoOfTaskSets);
            this.groupBox2.Controls.Add(this.label5);
            this.groupBox2.Location = new System.Drawing.Point(6, 102);
            this.groupBox2.Name = "groupBox2";
            this.groupBox2.Size = new System.Drawing.Size(744, 161);
            this.groupBox2.TabIndex = 44;
            this.groupBox2.TabStop = false;
            this.groupBox2.Text = "Task Generation Configuration";
            // 
            // checkBoxPruneFolder
            // 
            this.checkBoxPruneFolder.AutoSize = true;
            this.checkBoxPruneFolder.Checked = true;
            this.checkBoxPruneFolder.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkBoxPruneFolder.Location = new System.Drawing.Point(9, 69);
            this.checkBoxPruneFolder.Name = "checkBoxPruneFolder";
            this.checkBoxPruneFolder.Size = new System.Drawing.Size(224, 17);
            this.checkBoxPruneFolder.TabIndex = 63;
            this.checkBoxPruneFolder.Text = "Prune folder before creating task set file(s)";
            this.checkBoxPruneFolder.UseVisualStyleBackColor = true;
            // 
            // checkUniquePeriod
            // 
            this.checkUniquePeriod.AutoSize = true;
            this.checkUniquePeriod.Checked = true;
            this.checkUniquePeriod.CheckState = System.Windows.Forms.CheckState.Checked;
            this.checkUniquePeriod.Location = new System.Drawing.Point(8, 92);
            this.checkUniquePeriod.Name = "checkUniquePeriod";
            this.checkUniquePeriod.Size = new System.Drawing.Size(152, 17);
            this.checkUniquePeriod.TabIndex = 62;
            this.checkUniquePeriod.Text = "Unique periods in each set";
            this.checkUniquePeriod.UseVisualStyleBackColor = true;
            // 
            // checkUniqueExecutionTime
            // 
            this.checkUniqueExecutionTime.AutoSize = true;
            this.checkUniqueExecutionTime.Location = new System.Drawing.Point(8, 114);
            this.checkUniqueExecutionTime.Name = "checkUniqueExecutionTime";
            this.checkUniqueExecutionTime.Size = new System.Drawing.Size(194, 17);
            this.checkUniqueExecutionTime.TabIndex = 61;
            this.checkUniqueExecutionTime.Text = "Unique Execution Time in Each Set";
            this.checkUniqueExecutionTime.UseVisualStyleBackColor = true;
            // 
            // textMinimumOffset
            // 
            this.textMinimumOffset.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textMinimumOffset.Location = new System.Drawing.Point(679, 112);
            this.textMinimumOffset.Name = "textMinimumOffset";
            this.textMinimumOffset.Size = new System.Drawing.Size(56, 20);
            this.textMinimumOffset.TabIndex = 60;
            this.textMinimumOffset.Text = "0";
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(379, 115);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(168, 13);
            this.label7.TabIndex = 59;
            this.label7.Text = "MINimum value of Release Offset ";
            // 
            // textMaximumOffset
            // 
            this.textMaximumOffset.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textMaximumOffset.Location = new System.Drawing.Point(679, 138);
            this.textMaximumOffset.Name = "textMaximumOffset";
            this.textMaximumOffset.Size = new System.Drawing.Size(56, 20);
            this.textMaximumOffset.TabIndex = 58;
            this.textMaximumOffset.Text = "3";
            // 
            // label8
            // 
            this.label8.AutoSize = true;
            this.label8.Location = new System.Drawing.Point(379, 141);
            this.label8.Name = "label8";
            this.label8.Size = new System.Drawing.Size(168, 13);
            this.label8.TabIndex = 57;
            this.label8.Text = "MAXimum value of Release Offset";
            // 
            // checkRunSimulation
            // 
            this.checkRunSimulation.AutoSize = true;
            this.checkRunSimulation.Location = new System.Drawing.Point(8, 137);
            this.checkRunSimulation.Name = "checkRunSimulation";
            this.checkRunSimulation.Size = new System.Drawing.Size(217, 17);
            this.checkRunSimulation.TabIndex = 56;
            this.checkRunSimulation.Text = "Run simulation after generating task sets";
            this.checkRunSimulation.UseVisualStyleBackColor = true;
            // 
            // textMinimumPeriod
            // 
            this.textMinimumPeriod.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textMinimumPeriod.Location = new System.Drawing.Point(679, 12);
            this.textMinimumPeriod.Name = "textMinimumPeriod";
            this.textMinimumPeriod.Size = new System.Drawing.Size(56, 20);
            this.textMinimumPeriod.TabIndex = 55;
            this.textMinimumPeriod.Text = "40";
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(379, 15);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(163, 13);
            this.label4.TabIndex = 54;
            this.label4.Text = "MINimum value for Arrival Period ";
            // 
            // textMinimumExecutionTime
            // 
            this.textMinimumExecutionTime.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textMinimumExecutionTime.Location = new System.Drawing.Point(679, 63);
            this.textMinimumExecutionTime.Name = "textMinimumExecutionTime";
            this.textMinimumExecutionTime.Size = new System.Drawing.Size(56, 20);
            this.textMinimumExecutionTime.TabIndex = 53;
            this.textMinimumExecutionTime.Text = "2";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(379, 66);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(167, 13);
            this.label2.TabIndex = 52;
            this.label2.Text = "MINimum value of Execution time ";
            // 
            // textNoOfTasks
            // 
            this.textNoOfTasks.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textNoOfTasks.Location = new System.Drawing.Point(305, 13);
            this.textNoOfTasks.Name = "textNoOfTasks";
            this.textNoOfTasks.Size = new System.Drawing.Size(56, 20);
            this.textNoOfTasks.TabIndex = 51;
            this.textNoOfTasks.Text = "6";
            // 
            // label10
            // 
            this.label10.AutoSize = true;
            this.label10.Location = new System.Drawing.Point(5, 19);
            this.label10.Name = "label10";
            this.label10.Size = new System.Drawing.Size(106, 13);
            this.label10.TabIndex = 50;
            this.label10.Text = "# Tasks in each  Set";
            // 
            // textMaximumExecutionTime
            // 
            this.textMaximumExecutionTime.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textMaximumExecutionTime.Location = new System.Drawing.Point(679, 89);
            this.textMaximumExecutionTime.Name = "textMaximumExecutionTime";
            this.textMaximumExecutionTime.Size = new System.Drawing.Size(56, 20);
            this.textMaximumExecutionTime.TabIndex = 39;
            this.textMaximumExecutionTime.Text = "7";
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(379, 92);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(167, 13);
            this.label6.TabIndex = 38;
            this.label6.Text = "MAXimum value of Execution time";
            // 
            // textMaximumPeriod
            // 
            this.textMaximumPeriod.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textMaximumPeriod.Location = new System.Drawing.Point(679, 39);
            this.textMaximumPeriod.Name = "textMaximumPeriod";
            this.textMaximumPeriod.Size = new System.Drawing.Size(56, 20);
            this.textMaximumPeriod.TabIndex = 37;
            this.textMaximumPeriod.Text = "70";
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(379, 42);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(163, 13);
            this.label3.TabIndex = 36;
            this.label3.Text = "MAXimum value for Arrival Period";
            // 
            // textNoOfTaskSets
            // 
            this.textNoOfTaskSets.BackColor = System.Drawing.Color.FromArgb(((int)(((byte)(255)))), ((int)(((byte)(224)))), ((int)(((byte)(192)))));
            this.textNoOfTaskSets.Location = new System.Drawing.Point(305, 37);
            this.textNoOfTaskSets.Name = "textNoOfTaskSets";
            this.textNoOfTaskSets.Size = new System.Drawing.Size(56, 20);
            this.textNoOfTaskSets.TabIndex = 35;
            this.textNoOfTaskSets.Text = "200";
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(5, 40);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(148, 13);
            this.label5.TabIndex = 34;
            this.label5.Text = "# Task Sets to be Generated ";
            // 
            // menuStrip1
            // 
            this.menuStrip1.Items.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.fileToolStripMenuItem,
            this.optionsToolStripMenuItem,
            this.toolsToolStripMenuItem,
            this.aboutToolStripMenuItem});
            this.menuStrip1.Location = new System.Drawing.Point(0, 0);
            this.menuStrip1.Name = "menuStrip1";
            this.menuStrip1.Size = new System.Drawing.Size(758, 24);
            this.menuStrip1.TabIndex = 67;
            this.menuStrip1.Text = "menuStrip1";
            // 
            // fileToolStripMenuItem
            // 
            this.fileToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.loadConfigurationToolStripMenuItem,
            this.saveConfigurationToolStripMenuItem});
            this.fileToolStripMenuItem.Name = "fileToolStripMenuItem";
            this.fileToolStripMenuItem.Size = new System.Drawing.Size(37, 20);
            this.fileToolStripMenuItem.Text = "&File";
            // 
            // loadConfigurationToolStripMenuItem
            // 
            this.loadConfigurationToolStripMenuItem.Name = "loadConfigurationToolStripMenuItem";
            this.loadConfigurationToolStripMenuItem.Size = new System.Drawing.Size(177, 22);
            this.loadConfigurationToolStripMenuItem.Text = "&Load Configuration";
            this.loadConfigurationToolStripMenuItem.Click += new System.EventHandler(this.loadConfigurationToolStripMenuItem_Click);
            // 
            // saveConfigurationToolStripMenuItem
            // 
            this.saveConfigurationToolStripMenuItem.Name = "saveConfigurationToolStripMenuItem";
            this.saveConfigurationToolStripMenuItem.Size = new System.Drawing.Size(177, 22);
            this.saveConfigurationToolStripMenuItem.Text = "&Save Configuration";
            this.saveConfigurationToolStripMenuItem.Click += new System.EventHandler(this.saveConfigurationToolStripMenuItem_Click);
            // 
            // optionsToolStripMenuItem
            // 
            this.optionsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.setPeriodGeneratorToolStripMenuItem,
            this.setExecutionTimeGeneratorToolStripMenuItem,
            this.setRelToolStripMenuItem,
            this.setTaskSetGeneratorToolStripMenuItem,
            this.toolStripSeparator1,
            this.setExecutionModelToolStripMenuItem,
            this.setExecutionTraceWriterToolStripMenuItem});
            this.optionsToolStripMenuItem.Name = "optionsToolStripMenuItem";
            this.optionsToolStripMenuItem.Size = new System.Drawing.Size(61, 20);
            this.optionsToolStripMenuItem.Text = "&Options";
            // 
            // setPeriodGeneratorToolStripMenuItem
            // 
            this.setPeriodGeneratorToolStripMenuItem.Name = "setPeriodGeneratorToolStripMenuItem";
            this.setPeriodGeneratorToolStripMenuItem.Size = new System.Drawing.Size(229, 22);
            this.setPeriodGeneratorToolStripMenuItem.Text = "Set &Period Generator";
            this.setPeriodGeneratorToolStripMenuItem.Click += new System.EventHandler(this.setPeriodGeneratorToolStripMenuItem_Click);
            // 
            // setExecutionTimeGeneratorToolStripMenuItem
            // 
            this.setExecutionTimeGeneratorToolStripMenuItem.Name = "setExecutionTimeGeneratorToolStripMenuItem";
            this.setExecutionTimeGeneratorToolStripMenuItem.Size = new System.Drawing.Size(229, 22);
            this.setExecutionTimeGeneratorToolStripMenuItem.Text = "Set &Execution Time Generator";
            this.setExecutionTimeGeneratorToolStripMenuItem.Click += new System.EventHandler(this.setExecutionTimeGeneratorToolStripMenuItem_Click);
            // 
            // setRelToolStripMenuItem
            // 
            this.setRelToolStripMenuItem.Name = "setRelToolStripMenuItem";
            this.setRelToolStripMenuItem.Size = new System.Drawing.Size(229, 22);
            this.setRelToolStripMenuItem.Text = "Set &Release Offset Generator";
            this.setRelToolStripMenuItem.Click += new System.EventHandler(this.setRelToolStripMenuItem_Click);
            // 
            // setTaskSetGeneratorToolStripMenuItem
            // 
            this.setTaskSetGeneratorToolStripMenuItem.Name = "setTaskSetGeneratorToolStripMenuItem";
            this.setTaskSetGeneratorToolStripMenuItem.Size = new System.Drawing.Size(229, 22);
            this.setTaskSetGeneratorToolStripMenuItem.Text = "Set Task Set Generator";
            this.setTaskSetGeneratorToolStripMenuItem.Click += new System.EventHandler(this.setTaskSetGeneratorToolStripMenuItem_Click);
            // 
            // toolStripSeparator1
            // 
            this.toolStripSeparator1.Name = "toolStripSeparator1";
            this.toolStripSeparator1.Size = new System.Drawing.Size(226, 6);
            // 
            // setExecutionModelToolStripMenuItem
            // 
            this.setExecutionModelToolStripMenuItem.Name = "setExecutionModelToolStripMenuItem";
            this.setExecutionModelToolStripMenuItem.Size = new System.Drawing.Size(229, 22);
            this.setExecutionModelToolStripMenuItem.Text = "Set Execution Model";
            this.setExecutionModelToolStripMenuItem.Click += new System.EventHandler(this.setExecutionModelToolStripMenuItem_Click);
            // 
            // setExecutionTraceWriterToolStripMenuItem
            // 
            this.setExecutionTraceWriterToolStripMenuItem.Name = "setExecutionTraceWriterToolStripMenuItem";
            this.setExecutionTraceWriterToolStripMenuItem.Size = new System.Drawing.Size(229, 22);
            this.setExecutionTraceWriterToolStripMenuItem.Text = "Set Execution Trace Writer";
            this.setExecutionTraceWriterToolStripMenuItem.Click += new System.EventHandler(this.setExecutionTraceWriterToolStripMenuItem_Click);
            // 
            // toolsToolStripMenuItem
            // 
            this.toolsToolStripMenuItem.DropDownItems.AddRange(new System.Windows.Forms.ToolStripItem[] {
            this.simulationToolStripMenuItem});
            this.toolsToolStripMenuItem.Name = "toolsToolStripMenuItem";
            this.toolsToolStripMenuItem.Size = new System.Drawing.Size(48, 20);
            this.toolsToolStripMenuItem.Text = "&Tools";
            // 
            // simulationToolStripMenuItem
            // 
            this.simulationToolStripMenuItem.Name = "simulationToolStripMenuItem";
            this.simulationToolStripMenuItem.Size = new System.Drawing.Size(131, 22);
            this.simulationToolStripMenuItem.Text = "&Simulation";
            this.simulationToolStripMenuItem.Click += new System.EventHandler(this.simulationToolStripMenuItem_Click);
            // 
            // aboutToolStripMenuItem
            // 
            this.aboutToolStripMenuItem.Name = "aboutToolStripMenuItem";
            this.aboutToolStripMenuItem.Size = new System.Drawing.Size(52, 20);
            this.aboutToolStripMenuItem.Text = "&About";
            this.aboutToolStripMenuItem.Click += new System.EventHandler(this.aboutToolStripMenuItem_Click);
            // 
            // btnStop
            // 
            this.btnStop.Location = new System.Drawing.Point(596, 632);
            this.btnStop.Name = "btnStop";
            this.btnStop.Size = new System.Drawing.Size(75, 27);
            this.btnStop.TabIndex = 69;
            this.btnStop.Text = "S&top";
            this.btnStop.UseVisualStyleBackColor = true;
            this.btnStop.Click += new System.EventHandler(this.btnStop_Click);
            // 
            // btnStart
            // 
            this.btnStart.Location = new System.Drawing.Point(678, 632);
            this.btnStart.Name = "btnStart";
            this.btnStart.Size = new System.Drawing.Size(75, 27);
            this.btnStart.TabIndex = 68;
            this.btnStart.Text = "&Start";
            this.btnStart.UseVisualStyleBackColor = true;
            this.btnStart.Click += new System.EventHandler(this.btnStart_Click);
            // 
            // groupBox3
            // 
            this.groupBox3.Controls.Add(this.checkedListBoxSchedulabilityTests);
            this.groupBox3.Location = new System.Drawing.Point(6, 269);
            this.groupBox3.Name = "groupBox3";
            this.groupBox3.Size = new System.Drawing.Size(744, 78);
            this.groupBox3.TabIndex = 70;
            this.groupBox3.TabStop = false;
            this.groupBox3.Text = "Schedulability Tests";
            // 
            // checkedListBoxSchedulabilityTests
            // 
            this.checkedListBoxSchedulabilityTests.FormattingEnabled = true;
            this.checkedListBoxSchedulabilityTests.HorizontalScrollbar = true;
            this.checkedListBoxSchedulabilityTests.Location = new System.Drawing.Point(6, 19);
            this.checkedListBoxSchedulabilityTests.Name = "checkedListBoxSchedulabilityTests";
            this.checkedListBoxSchedulabilityTests.Size = new System.Drawing.Size(733, 49);
            this.checkedListBoxSchedulabilityTests.TabIndex = 0;
            // 
            // groupBox4
            // 
            this.groupBox4.Controls.Add(this.checkedListBoxSatisfiabilityConditions);
            this.groupBox4.Location = new System.Drawing.Point(6, 353);
            this.groupBox4.Name = "groupBox4";
            this.groupBox4.Size = new System.Drawing.Size(744, 78);
            this.groupBox4.TabIndex = 71;
            this.groupBox4.TabStop = false;
            this.groupBox4.Text = "Satisfiability Conditions ";
            // 
            // checkedListBoxSatisfiabilityConditions
            // 
            this.checkedListBoxSatisfiabilityConditions.FormattingEnabled = true;
            this.checkedListBoxSatisfiabilityConditions.HorizontalScrollbar = true;
            this.checkedListBoxSatisfiabilityConditions.Location = new System.Drawing.Point(6, 19);
            this.checkedListBoxSatisfiabilityConditions.Name = "checkedListBoxSatisfiabilityConditions";
            this.checkedListBoxSatisfiabilityConditions.Size = new System.Drawing.Size(732, 49);
            this.checkedListBoxSatisfiabilityConditions.TabIndex = 0;
            
            // 
            // groupBox5
            // 
            this.groupBox5.Controls.Add(this.checkedListBoxFileFormatter);
            this.groupBox5.Location = new System.Drawing.Point(6, 437);
            this.groupBox5.Name = "groupBox5";
            this.groupBox5.Size = new System.Drawing.Size(744, 78);
            this.groupBox5.TabIndex = 72;
            this.groupBox5.TabStop = false;
            this.groupBox5.Text = "File Formatter";
            // 
            // checkedListBoxFileFormatter
            // 
            this.checkedListBoxFileFormatter.FormattingEnabled = true;
            this.checkedListBoxFileFormatter.HorizontalScrollbar = true;
            this.checkedListBoxFileFormatter.Location = new System.Drawing.Point(6, 19);
            this.checkedListBoxFileFormatter.Name = "checkedListBoxFileFormatter";
            this.checkedListBoxFileFormatter.Size = new System.Drawing.Size(732, 49);
            this.checkedListBoxFileFormatter.TabIndex = 0;
            // 
            // groupBox6
            // 
            this.groupBox6.Controls.Add(this.textBoxMessages);
            this.groupBox6.Location = new System.Drawing.Point(6, 521);
            this.groupBox6.Name = "groupBox6";
            this.groupBox6.Size = new System.Drawing.Size(744, 105);
            this.groupBox6.TabIndex = 73;
            this.groupBox6.TabStop = false;
            this.groupBox6.Text = "Run-time Messages";
            // 
            // textBoxMessages
            // 
            this.textBoxMessages.Location = new System.Drawing.Point(9, 17);
            this.textBoxMessages.Multiline = true;
            this.textBoxMessages.Name = "textBoxMessages";
            this.textBoxMessages.ScrollBars = System.Windows.Forms.ScrollBars.Both;
            this.textBoxMessages.Size = new System.Drawing.Size(730, 82);
            this.textBoxMessages.TabIndex = 0;
            this.textBoxMessages.WordWrap = false;
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Font = new System.Drawing.Font("Microsoft Sans Serif", 12F, System.Drawing.FontStyle.Bold, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.label1.ForeColor = System.Drawing.Color.Blue;
            this.label1.Location = new System.Drawing.Point(4, 634);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(85, 20);
            this.label1.TabIndex = 74;
            this.label1.Text = "TGSIMEx";
            // 
            // fMain
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(758, 684);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.groupBox6);
            this.Controls.Add(this.groupBox5);
            this.Controls.Add(this.groupBox2);
            this.Controls.Add(this.groupBox4);
            this.Controls.Add(this.groupBox3);
            this.Controls.Add(this.btnStop);
            this.Controls.Add(this.btnStart);
            this.Controls.Add(this.statusStrip1);
            this.Controls.Add(this.menuStrip1);
            this.Controls.Add(this.groupBox1);
            this.FormBorderStyle = System.Windows.Forms.FormBorderStyle.Fixed3D;
            this.MainMenuStrip = this.menuStrip1;
            this.MaximizeBox = false;
            this.Name = "fMain";
            this.StartPosition = System.Windows.Forms.FormStartPosition.CenterScreen;
            this.Text = "Task Generation and Simulation - Extensible";
            this.Load += new System.EventHandler(this.fMain_Load);
            this.groupBox1.ResumeLayout(false);
            this.groupBox1.PerformLayout();
            this.statusStrip1.ResumeLayout(false);
            this.statusStrip1.PerformLayout();
            this.groupBox2.ResumeLayout(false);
            this.groupBox2.PerformLayout();
            this.menuStrip1.ResumeLayout(false);
            this.menuStrip1.PerformLayout();
            this.groupBox3.ResumeLayout(false);
            this.groupBox4.ResumeLayout(false);
            this.groupBox5.ResumeLayout(false);
            this.groupBox6.ResumeLayout(false);
            this.groupBox6.PerformLayout();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.GroupBox groupBox1;
        private System.Windows.Forms.Button btnBrowse;
        private System.Windows.Forms.TextBox textFolderName;
        private System.Windows.Forms.StatusStrip statusStrip1;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLabel1;
        private System.Windows.Forms.GroupBox groupBox2;
        private System.Windows.Forms.TextBox textMinimumPeriod;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.TextBox textMinimumExecutionTime;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.TextBox textNoOfTasks;
        private System.Windows.Forms.Label label10;
        private System.Windows.Forms.TextBox textMaximumExecutionTime;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.TextBox textMaximumPeriod;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.TextBox textNoOfTaskSets;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.FolderBrowserDialog folderBrowserDialog1;
        private System.Windows.Forms.MenuStrip menuStrip1;
        private System.Windows.Forms.ToolStripMenuItem optionsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem setPeriodGeneratorToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem setExecutionTimeGeneratorToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem setRelToolStripMenuItem;
        private System.Windows.Forms.Button btnStop;
        private System.Windows.Forms.Button btnStart;
        private System.Windows.Forms.GroupBox groupBox3;
        private System.Windows.Forms.CheckedListBox checkedListBoxSchedulabilityTests;
        private System.Windows.Forms.GroupBox groupBox4;
        private System.Windows.Forms.CheckedListBox checkedListBoxSatisfiabilityConditions;
        private System.Windows.Forms.GroupBox groupBox5;
        private System.Windows.Forms.CheckedListBox checkedListBoxFileFormatter;
        private System.Windows.Forms.GroupBox groupBox6;
        private System.Windows.Forms.TextBox textBoxMessages;
        private System.Windows.Forms.ToolStripMenuItem setExecutionModelToolStripMenuItem;
        private System.Windows.Forms.CheckBox checkRunSimulation;
        private System.Windows.Forms.TextBox textMinimumOffset;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.TextBox textMaximumOffset;
        private System.Windows.Forms.Label label8;
        private System.Windows.Forms.ToolStripStatusLabel toolStripStatusLeft;
        private System.Windows.Forms.CheckBox checkUniquePeriod;
        private System.Windows.Forms.CheckBox checkUniqueExecutionTime;
        private System.Windows.Forms.ToolStripMenuItem aboutToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem toolsToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem simulationToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem fileToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem loadConfigurationToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem saveConfigurationToolStripMenuItem;
        private System.Windows.Forms.ToolStripMenuItem setTaskSetGeneratorToolStripMenuItem;
        private System.Windows.Forms.ToolStripSeparator toolStripSeparator1;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.ToolStripMenuItem setExecutionTraceWriterToolStripMenuItem;
        private System.Windows.Forms.CheckBox checkBoxPruneFolder;
    }
}

