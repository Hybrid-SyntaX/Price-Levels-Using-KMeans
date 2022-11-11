using Microsoft.ML.Data;

namespace LevelsFinder;
public class FloatData
{
    public FloatData(float value)
    {
        Value = value;
    }

    [LoadColumn(0)]
    public float Value { get; }

    public static implicit operator FloatData(float value)
    {
        return new FloatData(value);
    }

    public static IEnumerable<FloatData> FromFloat(float[] floats)
    {
        foreach (var x in floats)
            yield return new FloatData(x);
    }

}

