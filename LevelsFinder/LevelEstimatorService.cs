using Kneedle;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using System.Reflection.Emit;

namespace LevelsFinder;

public class LevelEstimatorService
{
    private readonly MLContext _mlContext;
    private readonly int _maxKee;

    public LevelEstimatorService(int? seed, int maxKee = 10)
    {
        this._mlContext = new MLContext(seed);
        _maxKee = maxKee;
    }

    public TransformerChain<ClusteringPredictionTransformer<KMeansModelParameters>> Cluster(float[] data, int numberOfClusters = 3)
    {

        numberOfClusters = OptimizeNumberOfClusters(data.Length, numberOfClusters);

        IDataView dataView = _mlContext.Data.LoadFromEnumerable(FloatData.FromFloat(data));

        var options = new KMeansTrainer.Options
        {
            InitializationAlgorithm = KMeansTrainer.InitializationAlgorithm.KMeansYinyang,
            NumberOfClusters = numberOfClusters
        };

        var kmeans = _mlContext.Clustering.Trainers.KMeans(options);
        var pipeline = _mlContext.Transforms
                     .Concatenate("Features", "Value")
                     .Append(kmeans);

        return pipeline.Fit(dataView);

    }



    public IEnumerable<double> Elbow(float[] data, int maxK = 10)
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
            yield return result /= data.Length;
        }
    }
    public List<Level> MinMax(float[] data, int knee, int[] clusters)
    {
        List<Level> minMax = new();
        for (int i = 0; i <= knee; i++)
            minMax.Add(new Level { Min = float.PositiveInfinity, Max = float.NegativeInfinity });

        for (int i = 0; i < data.Length; i++)
        {
            var cluster = clusters[i];

            if (data[i] > minMax[cluster].Max)
                minMax[cluster].Max = data[i];
            if (data[i] < minMax[cluster].Min)
                minMax[cluster].Min = data[i];

        }
        minMax.Sort(new LevelComparer());

        return minMax.Where(x=>x.Min!=float.PositiveInfinity && x.Max !=float.NegativeInfinity).ToList();

    }
    public List<Level> FindLevels(float[] data)
    {
        if(data.Length < _maxKee)
        {
            throw new ApplicationException("Data length must be more than max knee value");
        }

        data = data.SkipWhile(x => double.IsNaN(x)).ToArray();

        var elbows = Elbow(data,_maxKee).ToArray();

        var k = Enumerable.Range(1, _maxKee).Select(x => (double)x).ToArray();
        var kneed = KneedleAlgorithm.CalculateKneePoints(k, elbows, CurveDirection.Decreasing, Curvature.Counterclockwise, forceLinearInterpolation: false);
        var clusterTransformer = Cluster(data, numberOfClusters: Convert.ToInt32(kneed));
        var predictor = _mlContext.Model.CreatePredictionEngine<FloatData, ClusterPrediction>(clusterTransformer);
        var preds = data.Select(x => predictor.Predict(x).PredictedClusterId).ToArray();
        return MinMax(data, Convert.ToInt32(kneed), preds.Select(x => (int)x).ToArray());
    }

    public IEnumerable<Level> FindLevels(double[] data)
    {
        return FindLevels(data.Select(x => (float)x).ToArray());
    }
    private static float CalculateDistance(float x, float y, int k = 2)
    => (float)Math.Sqrt(Math.Pow(x - y, k));

    /// <summary>
    /// A remedy against the times that examples are too few and we don't want algorithm to fail.
    /// </summary>
    /// <param name="dataLength"></param>
    /// <param name="numberOfClusters"></param>
    /// <returns></returns>
    private static int OptimizeNumberOfClusters(int dataLength, int numberOfClusters)
    {
        if (numberOfClusters == 0)
            return 1;

        if (dataLength / numberOfClusters > 0.09 * dataLength)
            return numberOfClusters;
        else
            return OptimizeNumberOfClusters(dataLength, numberOfClusters - 1);
    }
}

