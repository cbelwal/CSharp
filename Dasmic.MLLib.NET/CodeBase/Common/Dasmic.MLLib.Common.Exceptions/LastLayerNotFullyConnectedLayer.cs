using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Common.Exceptions
{
    public class LastLayerNotFullyConnectedLayer : Exception
    {
        public LastLayerNotFullyConnectedLayer(string message,
                    Exception innerException) : base(message, innerException)
        {

        }

        public LastLayerNotFullyConnectedLayer()
        {

        }

        public override string Message
        {
            get
            {
                return Resources.strings_messages.exception_last_layer_not_fully_connected;
            }
        }
    }
}
