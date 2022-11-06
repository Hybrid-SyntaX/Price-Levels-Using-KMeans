// See https://aka.ms/new-console-template for more information
using ClusteringTuts;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Collections;
using System.Collections.Generic;
using System.Reflection;
using Microsoft.ML.Trainers;
using System.Data;
using static Microsoft.ML.Data.SchemaDefinition;
using System.Security.Cryptography;
using Kneedle;

internal class Program
{
    static MLContext mlContext = new MLContext(seed: 0);

    public static float CalculateDistance(float x, float y, int k = 2)
        => (float)Math.Sqrt(Math.Pow(x - y, k));
    static TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>> Cluster(float[] data, int numberOfClusters = 3)
    {
        IDataView dataView = mlContext.Data.LoadFromEnumerable(FloatData.FromFloat(data));

        var kmeans = mlContext.Clustering.Trainers.KMeans(numberOfClusters: numberOfClusters);
        var pipeline = mlContext.Transforms
                     .Concatenate("Features", "Value")
                     .Append(kmeans);

        return pipeline.Fit(dataView);

    }

    static IEnumerable<double> Elbow(float[] data, int maxK = 10)
    {
        for (int k = 1; k <= maxK; k++)
        {

            var cluster = Cluster(data, numberOfClusters: k);
            VBuffer<float>[] centroids = new VBuffer<float>[] { };
            int centroidK = 0;
            cluster.LastTransformer.Model.GetClusterCentroids(ref centroids, out centroidK);


            var result = data.Sum(
                point => centroids.Select(x => x.DenseValues().First()).Min(
                    centroid => CalculateDistance(point, centroid)
                    )
                );
            yield return  result /= data.Length;
        }
    }
    class Level
    {
        public float Min { set; get; } = float.PositiveInfinity;
        public float Max { set; get; } = float.NegativeInfinity;
    }
    class LevelComparer : IComparer<Level>
    {
        public int Compare(Level? x, Level? y)
        {
            if (x.Min > y.Min)
                return 1;
            else if (x.Min < y.Min)
                return -1;
            else 
                return 0;
        }


    }
    static List<Level> MinMax(float[] data, int knee, int [] clusters)
    {
        const int MAX_IDX= 1;
        const int MIN_IDX= 0;
        List<Level> minMax = new ();
        for (int i = 0; i <= knee; i++)
            minMax.Add(new Level { Min= float.PositiveInfinity, Max= float.NegativeInfinity});

        for (int i=0; i < data.Length; i++)
        {
            var cluster = clusters[i];

            if (data[i] > minMax[cluster].Max)
                minMax[cluster].Max = data[i];
            if (data[i] < minMax[cluster].Min)
                minMax[cluster].Min = data[i];

        }
        minMax.Sort(new LevelComparer());
        return minMax;

    }
    private static void Main(string[] args)
    {
        Console.WriteLine("Hello, World!");



        float[] data = new float[] { 30.0f, 30.1f, 30.54f, 40f, 41, 56, 70, 71, 72, 69 };

        var elbows = Elbow(data).ToArray();

        var k = Enumerable.Range(1, 10).Select(x => (double)x).ToArray();
        var kneed = KneedleAlgorithm.CalculateKneePoints(k, elbows, CurveDirection.Decreasing, Curvature.Counterclockwise, forceLinearInterpolation: false);
        var clusterTransformer = Cluster(data, numberOfClusters: Convert.ToInt32(kneed));

        var predictor = mlContext.Model.CreatePredictionEngine<FloatData, ClusterPrediction>(clusterTransformer);
        var preds = data.Select(x => predictor.Predict(x).PredictedClusterId).ToArray();
        var minMax = MinMax(data, Convert.ToInt32(kneed), preds.Select(x=>(int)x).ToArray());
        foreach(var mm in minMax)
        {
            Console.WriteLine($"({mm.Min},{mm.Max})");
        }

        Console.ReadKey();
    }
}



