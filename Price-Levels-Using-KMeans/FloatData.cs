using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ClusteringTuts
{
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
            foreach(var x in floats)
                yield return new FloatData(x);
        }

    }

}
