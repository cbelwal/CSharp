using System;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class InvalidActivationFunctionException : Exception
    {

        public InvalidActivationFunctionException(
            string message,
            Exception innerException) : base(message, innerException)
        {

        }


        public InvalidActivationFunctionException()
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
