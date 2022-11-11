using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace LevelsFinder.Extensions
{
    public static class Extensions
    {
        public static string ToCommaSeperatedTuples(this List<Level> value)
        {
            var results = new string[value.Count];
            for (int i = 0; i < value.Count; i++)
            {
                results[i] = $"[{value[i].Min},{value[i].Max}]";

            }

            return $"[{string.Join(", ", results)}]";
        }
    }
}
