using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{

    public class InvalidPaddingValueException : Exception
    {
        public InvalidPaddingValueException(
            string message,
            Exception innerException) : base(message, innerException)
            {

        }

        public InvalidPaddingValueException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_cnn_padding_value;
            }
        }
    }
}
