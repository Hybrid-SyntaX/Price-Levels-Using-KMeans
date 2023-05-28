namespace LevelsFinder;

public class Level : IComparable<Level>, IComparer<Level>
{
    public float Min { set; get; } = float.PositiveInfinity;
    public float Max { set; get; } = float.NegativeInfinity;
    public float Mean => (Min + Max) / 2;

    public int Compare(Level? x, Level? y)
    {
        //return Math.Min(CompareMin(x, y),CompareMax(x,y));
        return CompareMean(x, y);
    }

    public int CompareTo(Level? other)
    {
        return Compare(this, other);
    }

    private int CompareMin(Level? x, Level? y)
    {
        if (y == null)
            return 1;

        if (x == null)
            return -1;

        if (x.Min > y.Min)
            return 1;

        if (x.Min < y.Min)
            return -1;

        return 0;
    }

    private int CompareMax(Level? x, Level? y)
    {
        if (y == null)
            return 1;

        if (x == null)
            return -1;

        if (x.Max > y.Max)
            return 1;

        if (x.Max < y.Max)
            return -1;

        return 0;
    }

    private int CompareMean(Level? x, Level? y)
    {
        if (y == null)
            return 1;

        if (x == null)
            return -1;

        if (x.Mean > y.Mean)
            return 1;

        if (x.Mean < y.Mean)
            return -1;

        return 0;
    }
    public override string ToString()
    {
        return $"({Min},{Max})";
    }
}



