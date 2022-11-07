// See https://aka.ms/new-console-template for more information
using ClusteringTuts;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Data;
using Kneedle;

internal partial class Program
{

    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");
        LevelEstimatorService levelEstimatorService = new LevelEstimatorService(new MLContext(seed: 0));


        float[] data = new float[] { 30.0f, 30.1f, 30.54f, 40f, 41, 56, 70, 71, 72, 69 };

        var minMax =levelEstimatorService.FindLevels(data);
        foreach (var mm in minMax)
        {
            Console.WriteLine($"({mm.Min},{mm.Max})");
        }

        Console.ReadKey();
    }
}



