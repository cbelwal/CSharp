using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace DemoAppConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Pythagoras pythagoras = new Pythagoras();
            Console.WriteLine("With 5 Samples ...");
            pythagoras.Run_5_Samples();
            Console.WriteLine();
            Console.WriteLine("With 10 Samples ...");
            pythagoras.Run_10_Samples();
            Console.WriteLine();
            Console.WriteLine("With 15 Samples ...");
            pythagoras.Run_15_Samples();
            Console.WriteLine();
            Console.WriteLine("With 15 Samples Neural Network ...");
            pythagoras.Run_15_Samples_NN();
            Console.ReadLine();
        }
    }
}
