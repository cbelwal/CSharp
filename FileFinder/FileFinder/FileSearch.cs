using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;

namespace FileFinder
{
    internal class FileSearch
    {
       
        public event EventHandler<string> FolderUpdate;
        public event EventHandler<string> FoundStringInFile;


        protected virtual void UpdateFolder(String folderName)
        {
            EventHandler<string> handler = FolderUpdate;
            if (handler != null)
            {
                handler(this, folderName);
            }
        }

        protected virtual void UpdateFoundStringInFile(String fileName)
        {
            EventHandler<string> handler = FoundStringInFile;
            if (handler != null)
            {
                handler(this, fileName);
            }
        }

        //Constructor should accept the searchText
        public FileSearch()
        {
            
        }

        public void SearchInFolder(string folderName, string searchText)
        {
            DirectoryInfo dirInfo = new DirectoryInfo(folderName);
            
            //Recursive for each folder
            foreach (DirectoryInfo di in dirInfo.EnumerateDirectories())
            {
                SearchInFolder(di.FullName,searchText);
            }

            UpdateFolder(dirInfo.FullName);

            //Then search through files
            foreach (FileInfo fi in dirInfo.EnumerateFiles())
            {
                
                SearchInFile(fi.FullName,searchText);
            }            

        }

        private void SearchInFile(string filePath,string searchText)
        {
            //Do a more optimized method later
            try
            {
                //StreamReader sr = File.OpenText(filePath);
                string fileData =  File.ReadAllText(filePath);
                if (fileData.Contains(searchText))
                    UpdateFoundStringInFile(filePath);
            }
            catch
            {

            }
            finally
            {

            }
            
        }
    }
}

