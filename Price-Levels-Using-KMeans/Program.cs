// See https://aka.ms/new-console-template for more information
using LevelsFinder;
using Microsoft.ML;


internal partial class Program
{

    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
        LevelEstimatorService levelEstimatorService = new LevelEstimatorService(new MLContext(seed: 0));


       var data = new double[] { 30.0f, 30.1f, 30.54f, 40f, 41, 56, 70, 71, 72, 69 };

        var minMax =levelEstimatorService.FindLevels(data);
        foreach (var mm in minMax)
        {
            Console.WriteLine($"({mm.Min},{mm.Max})");
        }

        Console.ReadKey();
    }
}



