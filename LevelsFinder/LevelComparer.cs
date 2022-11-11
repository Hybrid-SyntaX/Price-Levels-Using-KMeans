
namespace LevelsFinder;
public class LevelComparer : IComparer<Level>
{
    public int Compare(Level? x, Level? y)
    {
        if (x?.Min > y?.Min)
            return 1;
        else if (x?.Min < y?.Min)
            return -1;
        else
            return 0;
    }

}




