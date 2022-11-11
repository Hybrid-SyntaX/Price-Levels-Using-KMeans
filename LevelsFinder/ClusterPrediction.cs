using Microsoft.ML.Data;

namespace LevelsFinder;
public class ClusterPrediction
{
    [ColumnName("PredictedLabel")]
    public uint PredictedClusterId { set; get; }

    [ColumnName("Score")]
    public float[] Distances { set; get; }
}
