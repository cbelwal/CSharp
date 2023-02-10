using System;
using Dasmic.MLLib.Common.DataManagement;
using Dasmic.MLLib.Algorithms.Regression;
using Dasmic.MLLib.Algorithms.NeuralNetwork;


namespace DemoAppConsole
{
    internal class Pythagoras
    {
        public void Run_5_Samples()
        {
            string[] headers = { "Adjacent", "Opposite", "Hypotenuse" };
            DataSetCompact dsc = new DataSetCompact(headers, 2, -1);
            dsc.AddSingleRow(new double[] { 1, 1, 1.414213562 });
            dsc.AddSingleRow(new double[] { 1, 2, 2.236067977 });
            dsc.AddSingleRow(new double[] { 1, 3, 3.16227766 });
            dsc.AddSingleRow(new double[] { 2, 2, 2.828427125 });
            dsc.AddSingleRow(new double[] { 2, 3, 3.605551275 });

            Dasmic.MLLib.Common.MLCore.BuildBase build = new BuildLinearMultiVariable();
            Dasmic.MLLib.Common.MLCore.ModelBase model = build.BuildModel(dsc.GetAllDataRows(),
                                    dsc.GetAllAttributeHeaders(), dsc.GetIdxTargetAttribute());
            //Print the coeffs            
            Console.WriteLine(model.ToString());
            Console.WriteLine("RMSE:" + model.GetModelRMSE(dsc.GetAllDataRows()));

            dsc.AddSingleRow(new double[] { 2, 4, 4.472135955 });
            dsc.AddSingleRow(new double[] { 3, 3, 4.242640687 });
            dsc.AddSingleRow(new double[] { 3, 4, 5 });
            dsc.AddSingleRow(new double[] { 3, 5, 5.830951895 });
            dsc.AddSingleRow(new double[] { 3, 6, 6.708203932 });

            dsc.AddSingleRow(new double[] { 3, 7, 7.615773106 });
            dsc.AddSingleRow(new double[] { 4, 4, 5.656854249 });
            dsc.AddSingleRow(new double[] { 4, 5, 6.403124237 });
            dsc.AddSingleRow(new double[] { 4, 6, 7.211102551 });
            dsc.AddSingleRow(new double[] { 4, 7, 8.062257748 });

            PrintTable(dsc, model);            
        }

        public void Run_10_Samples()
        {
            string[] headers = { "Adjacent", "Opposite", "Hypotenuse" };
            DataSetCompact dsc = new DataSetCompact(headers, 2, -1);
            dsc.AddSingleRow(new double[] { 1, 1, 1.414213562 });
            dsc.AddSingleRow(new double[] { 1, 2, 2.236067977 });
            dsc.AddSingleRow(new double[] { 1, 3, 3.16227766 });
            dsc.AddSingleRow(new double[] { 2, 2, 2.828427125 });
            dsc.AddSingleRow(new double[] { 2, 3, 3.605551275 });

            dsc.AddSingleRow(new double[] { 2,4,4.472135955 });
            dsc.AddSingleRow(new double[] { 3,3,4.242640687 });
            dsc.AddSingleRow(new double[] { 3,4,5 });
            dsc.AddSingleRow(new double[] { 3,5,5.830951895 });
            dsc.AddSingleRow(new double[] { 3,6,6.708203932 });

            Dasmic.MLLib.Common.MLCore.BuildBase build = new BuildLinearMultiVariable();
            Dasmic.MLLib.Common.MLCore.ModelBase model = build.BuildModel(dsc.GetAllDataRows(),
                                    dsc.GetAllAttributeHeaders(), dsc.GetIdxTargetAttribute());

            //Print the coeffs            
            Console.WriteLine(model.ToString());
            Console.WriteLine("RMSE:"+ model.GetModelRMSE(dsc.GetAllDataRows()));

            dsc.AddSingleRow(new double[] { 3, 7, 7.615773106 });
            dsc.AddSingleRow(new double[] { 4, 4, 5.656854249 });
            dsc.AddSingleRow(new double[] { 4, 5, 6.403124237 });
            dsc.AddSingleRow(new double[] { 4, 6, 7.211102551 });
            dsc.AddSingleRow(new double[] { 4, 7, 8.062257748 });        

            PrintTable(dsc, model);

        }

        public void Run_15_Samples()
        {
            string[] headers = { "Adjacent", "Opposite", "Hypotenuse" };
            DataSetCompact dsc = new DataSetCompact(headers, 2, -1);
            dsc.AddSingleRow(new double[] { 1, 1, 1.414213562 });
            dsc.AddSingleRow(new double[] { 1, 2, 2.236067977 });
            dsc.AddSingleRow(new double[] { 1, 3, 3.16227766 });
            dsc.AddSingleRow(new double[] { 2, 2, 2.828427125 });
            dsc.AddSingleRow(new double[] { 2, 3, 3.605551275 });

            dsc.AddSingleRow(new double[] { 2, 4, 4.472135955 });
            dsc.AddSingleRow(new double[] { 3, 3, 4.242640687 });
            dsc.AddSingleRow(new double[] { 3, 4, 5 });
            dsc.AddSingleRow(new double[] { 3, 5, 5.830951895 });
            dsc.AddSingleRow(new double[] { 3, 6, 6.708203932 });

            dsc.AddSingleRow(new double[] { 3,7,7.615773106 });
            dsc.AddSingleRow(new double[] { 4,4,5.656854249 });
            dsc.AddSingleRow(new double[] { 4,5,6.403124237 });
            dsc.AddSingleRow(new double[] { 4,6,7.211102551 });
            dsc.AddSingleRow(new double[] { 4,7,8.062257748 });
           
            Dasmic.MLLib.Common.MLCore.BuildBase build = new BuildLinearMultiVariable();
            Dasmic.MLLib.Common.MLCore.ModelBase model = build.BuildModel(dsc.GetAllDataRows(),
                                    dsc.GetAllAttributeHeaders(), dsc.GetIdxTargetAttribute());

            //Print the coeffs            
            Console.WriteLine(model.ToString());
            Console.WriteLine("RMSE:" + model.GetModelRMSE(dsc.GetAllDataRows()));

            PrintTable(dsc, model);

        }

        public void Run_15_Samples_NN()
        {
            string[] headers = { "Adjacent", "Opposite", "Hypotenuse" };
            DataSetCompact dsc = new DataSetCompact(headers, 2, -1);
            dsc.AddSingleRow(new double[] { 1, 1, 1.414213562 });
            dsc.AddSingleRow(new double[] { 1, 2, 2.236067977 });
            dsc.AddSingleRow(new double[] { 1, 3, 3.16227766 });
            dsc.AddSingleRow(new double[] { 2, 2, 2.828427125 });
            dsc.AddSingleRow(new double[] { 2, 3, 3.605551275 });

            dsc.AddSingleRow(new double[] { 2, 4, 4.472135955 });
            dsc.AddSingleRow(new double[] { 3, 3, 4.242640687 });
            dsc.AddSingleRow(new double[] { 3, 4, 5 });
            dsc.AddSingleRow(new double[] { 3, 5, 5.830951895 });
            dsc.AddSingleRow(new double[] { 3, 6, 6.708203932 });

            dsc.AddSingleRow(new double[] { 3, 7, 7.615773106 });
            dsc.AddSingleRow(new double[] { 4, 4, 5.656854249 });
            dsc.AddSingleRow(new double[] { 4, 5, 6.403124237 });
            dsc.AddSingleRow(new double[] { 4, 6, 7.211102551 });
            dsc.AddSingleRow(new double[] { 4, 7, 8.062257748 });

            Build2LBackPropagation build = new Build2LBackPropagation();
            build.SetParameters(0,.0001, 10000, .001);
            //Change the activation function from Sigmoid
            build.SetActivationFunction(1, new Dasmic.MLLib.Algorithms.NeuralNetwork.Support.ActivationFunction.Linear());
            Dasmic.MLLib.Common.MLCore.ModelBase model = build.BuildModel(dsc.GetAllDataRows(),
                                    dsc.GetAllAttributeHeaders(), dsc.GetIdxTargetAttribute());

            //Print the coeffs            
            Console.WriteLine(model.ToString());
            //Print RMSE
            Console.WriteLine("RMSE:" + model.GetModelRMSE(dsc.GetAllDataRows()));

            PrintTable(dsc, model);

        }

        private void PrintTable(DataSetCompact dsc, Dasmic.MLLib.Common.MLCore.ModelBase model)
        {
            string origTable = "", finalTable = "";
            for (int idxRow = 0; idxRow < dsc.GetNumberOfDataRows(); idxRow++)
            {
                double[] data = dsc.GetSingleDataRowWithNoTargetValue(idxRow);
                double output = model.RunModelForSingleData(data);
                double target = dsc.GetTargetValue(idxRow);
                double percentageDiff = ((output - target) / target) * 100;

                origTable = origTable + "<tr>" + "<td>" + DataSetCompact.GetFormattedDouble(data[0], 3) + "</td>" +
                                     "<td>" + DataSetCompact.GetFormattedDouble(data[1], 3) + "</td>" +
                                     "<td>" + DataSetCompact.GetFormattedDouble(target, 3) + "</td>" + "</tr>" + "\r\n";

                finalTable = finalTable + "<tr>" + "<td>" + DataSetCompact.GetFormattedDouble(data[0], 3) + "</td>" +
                                     "<td>" + DataSetCompact.GetFormattedDouble(data[1], 3) + "</td>" +
                                     "<td>" + DataSetCompact.GetFormattedDouble(target, 3) + "</td>" +
                                     "<td>" + DataSetCompact.GetFormattedDouble(output, 3) + "</td>" +
                                     "<td>" + DataSetCompact.GetFormattedDouble(percentageDiff, 3) + "</td>" + "</tr>" + "\r\n";                
            }

            Console.WriteLine("Original Table");
            Console.WriteLine(origTable);
            Console.WriteLine("Final Table");
            Console.WriteLine(finalTable);
        }

    }
}
