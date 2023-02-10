using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Text;
using System.Reflection;

namespace TaskGen
{
    class CObjectFactory
    {
        public string errorMessages;
        /// <summary>
        /// Get List of Object that implement interfaceObj
        /// </summary>
        /// <param name="interfaceObj"></param>
        /// <returns></returns>
        public ArrayList GetObjects(Type typeObj, string appFolder)
        {
            // Get List of All Files
            ArrayList allObjects = new ArrayList();
            FileInfo[] dllFiles;

            try
            {
                DirectoryInfo di = new DirectoryInfo(appFolder + "\\PlugIns");
                dllFiles = di.GetFiles("*.dll");
            }
            catch
            {
                errorMessages = "Could not locate PlugIns";
                return null;
            }

            Assembly asm;
            Object tmpObject;


            errorMessages = "";

            foreach (FileInfo fi in dllFiles)
            {
                asm = Assembly.LoadFrom(fi.FullName);

                try
                {
                    if (asm.GetTypes() != null)
                    {
                        foreach (Type t in asm.GetTypes())
                        {
                            if (DoesImplementInterface(asm, t, typeObj))
                            {
                                tmpObject = asm.CreateInstance(t.ToString());
                                allObjects.Add(tmpObject);
                            }
                        }
                    }
                }
                catch(Exception e)
                {
                    errorMessages = errorMessages + "Could not load PlugIns from file " + 
                                fi.Name + " (System Message : "  + e.Message  +  ")\r\n";
                }

                // Check if they Implement Objects
            }
            return allObjects;

        }

        /// <summary>
        /// 
        /// </summary>
        /// <param name="asm"></param>
        /// <param name="t"></param>
        /// <param name="typeObj"></param>
        /// <returns></returns>
        private bool DoesImplementInterface(Assembly asm, Type t, Type typeObj)
        {
            
            if (t.GetInterface(typeObj.FullName) != null) return true;
            else return false;

        }


       

    }
}
