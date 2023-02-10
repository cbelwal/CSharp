using System;
using System.Windows.Input;
using System.Windows.Forms;
using System.ComponentModel;
using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;

namespace FileFinder
{
    public class ViewModelMain : INotifyPropertyChanged
    {
        private String _folderName;
        private String _searchText;
        private FileSearch _fileSearch;
        private string _statusMessage;
        private ObservableCollection<string> _foundFiles;    

        public event PropertyChangedEventHandler PropertyChanged;
        private readonly BackgroundWorker _worker = new BackgroundWorker();


        public string StatusMessage {
            get
            {
                return _statusMessage;
            }
            set
            {
                _statusMessage = value;
                NotifyPropertyChanged();
               
            }
        
        }
        public ObservableCollection<string> FoundFiles
        {
            get
            {
                return _foundFiles;
            }
            private set
            {
                _foundFiles = value;
            }
        }

        public ICommand StartCommand { get; set; }
        public ICommand BrowseCommand { get; set; }


        // This method is called by the Set accessor of each property.  
        // The CallerMemberName attribute that is applied to the optional propertyName  
        // parameter causes the property name of the caller to be substituted as an argument.  
        private void NotifyPropertyChanged([CallerMemberName] String propertyName = "")
        {
            PropertyChanged?.Invoke(this, new PropertyChangedEventArgs(propertyName));
        }

        public ViewModelMain()
        {
            StartCommand = new RelayCommand(OnStartClicked);
            BrowseCommand = new RelayCommand(OnBrowseClicked);
            FolderName = AppSettings.Default.FolderName;
            SearchText = AppSettings.Default.SearchText;
            FoundFiles = new ObservableCollection<string>();
            StatusMessage = Resources.StringResources.Ready;
            _fileSearch = new FileSearch();
            _fileSearch.FolderUpdate += _fileSearch_FolderUpdate;
            _fileSearch.FoundStringInFile += _fileSearch_FoundStringInFile;

            _worker.DoWork += worker_DoWork;
            _worker.RunWorkerCompleted += worker_RunWorkerCompleted;
        }

        private void _fileSearch_FoundStringInFile(object sender, string e)
        {
            AddToFoundFiles(e);
        }

        private void _fileSearch_FolderUpdate(object sender, string e)
        {
            StatusMessage = Resources.StringResources.SearchingFolder + e;                     
        }

        public String FolderName
        {
            get
            {
                return _folderName;
            }
            set
            {
                _folderName = value;
                NotifyPropertyChanged("FolderName");
                AppSettings.Default.FolderName = _folderName;
                AppSettings.Default.Save();
            }
        }

        /// <summary>
        /// Text to search
        /// </summary>
        public String SearchText
        {
            get
            {
                return _searchText;
            }
            set
            {
                _searchText = value;
                NotifyPropertyChanged("SearchText");
                AppSettings.Default.SearchText = _searchText;
                AppSettings.Default.Save();
            }
        }

      

        public void OnStartClicked(object obj)
        {
            //Start the Search  

            if (!_worker.IsBusy)
            {
                FoundFiles.Clear();
                NotifyPropertyChanged("FoundFiles");
                _worker.RunWorkerAsync();
            }
            //Execute in separate Thread           
        }

        public void OnBrowseClicked(object obj)
        {
            var dlg = new FolderBrowserDialog();
            DialogResult result = dlg.ShowDialog();
            FolderName = dlg.SelectedPath;            
        }

        private void worker_DoWork(object sender, DoWorkEventArgs e)
        {
            // run all background tasks here            
            StatusMessage = Resources.StringResources.Starting;
            AddStatusMessageToFoundFiles(StatusMessage);
            _fileSearch.SearchInFolder(_folderName, _searchText);
        }

        private void worker_RunWorkerCompleted(object sender,
                                                   RunWorkerCompletedEventArgs e)
        {
            //update ui once worker complete his work
            StatusMessage = Resources.StringResources.Completed;
            AddStatusMessageToFoundFiles(StatusMessage);
        }

        private void AddStatusMessageToFoundFiles(string message)
        {
            AddToFoundFiles("***" + message);
        }

        private void AddToFoundFiles(string message)
        {
            App.Current.Dispatcher.Invoke((Action)delegate // Invoker to run in UI Thread
            {
                FoundFiles.Add(message);
            });
            NotifyPropertyChanged("FoundFiles");
        }



    }
}
