using System;
using Dasmic.Core.License;
using System.Reflection;


namespace ExcelAddIn.Support
{
    public class LicenseInfo : ILicenseInfo, IDisposable
    {
        private String _appId;
        private BasicPermanentLicense _bpl;

        public String GetLicenseID()
        {
            return AppId + ":" + _bpl.GetSystemId();
        }

        public void Dispose()
        {
            _bpl = null;
        }

            
        public BasicPermanentLicense bpl
        {
            get
            {
                return _bpl;
            }
        }

        private String AppId
        {
            get
            {
                return _appId;
            }
        }

        private BasicPermanentLicense
            GetBasicPermanentLicense()
        {          
            //NOTE: Salt is also stored in Properties to confuse a hacker
            //That Salt can also be seen in Coju.exe.config.
            String salt = Properties.Resources.Value;
            BasicPermanentLicense bpl =
                new BasicPermanentLicense(AppId, salt);
            bpl.GetSystemId();
            return bpl;
        }
        

        public LicenseInfo()
        {
            _appId = Assembly.GetExecutingAssembly().GetType().GUID.ToString();
            _bpl = GetBasicPermanentLicense();
        }


        internal bool HasLicense( )
        {            
            Boolean flag;
            try
            {
                flag = bpl.CheckLicense();
            }
            catch
            {
                flag = false;
            }
            finally
            {
               
            }
            return flag;
        }

    }
}
