using System;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class InvalidPoolingFunctionException : Exception
    {

        public InvalidPoolingFunctionException(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public InvalidPoolingFunctionException()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_invalid_activation_function;
            }
        }
    }
}
