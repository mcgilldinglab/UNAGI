package edu.cmu.cs.sb.chromviewer;

import edu.cmu.cs.sb.core.*;
import java.util.*;

/**
 * Encapsulates information about a gene
 */
public class Gene 
{
    public String namechrom;
    
    public int chromosome;
    public char strand;

    public int start;
    public int end;

    /**
     * Converts the gene information to a unique integer hash code
     */
    public int hashCode()
    {
       String sz = start+"|"+end+"|"+
	    chromosome +"|"+strand+"|"+namechrom;
       return sz.hashCode();
    }

    /**
     * True if both objects are equal with all the same field values
     */
    public boolean equals(Object obj)
    {
       if (!(obj instanceof Gene))
       {
          return false;
       }
       else
       {
          Gene g = (Gene) obj;
	  boolean bsame = 
	    ((start == g.start) &&
	    (end == g.end) &&
            (chromosome==g.chromosome)&&
	    (strand==g.strand)&&
	    (((namechrom==null)&&(g.namechrom==null))||
	     ((namechrom!=null)&&(g.namechrom!=null)&&
	      namechrom.equals(g.namechrom))));
	
	  return bsame;
       }
    }
}
