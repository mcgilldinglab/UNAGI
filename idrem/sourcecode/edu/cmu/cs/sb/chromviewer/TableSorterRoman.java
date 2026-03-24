package edu.cmu.cs.sb.chromviewer;

import edu.cmu.cs.sb.core.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.util.List;

import javax.swing.*;
import javax.swing.event.TableModelEvent;
import javax.swing.event.TableModelListener;
import javax.swing.table.*;
import java.io.*;

/**
 * A table sorter for sorting chromosome identifiers
 */
public class TableSorterRoman extends TableSorter 
{

    public static final Comparator LEXICALROMAN_COMPARATOR = new ChromIDRecCompare();
    private int nromancol = -1;

    static int TYPE_ROMAN = 0;
    static int TYPE_NUMERIC = 1;
    static int TYPE_STRING = 2;

    /**
     * Calls empty constructor of parent class
     */
    public TableSorterRoman() 
    {
	super();
    }

    /**
     * Calls parent class constructor with tableModel; stores nromancol
     */
    public TableSorterRoman(TableModel tableModel,int nromancol) 
    {
        super(tableModel);
	this.nromancol = nromancol;
    }

    /**
     * Passes parameters to parent class
     */
    public TableSorterRoman(TableModel tableModel, JTableHeader tableHeader) 
    {
	super(tableModel,tableHeader);
    }



    /**
     * Class gives an ordering on chromosome names
     */
    static class ChromIDRecCompare implements Comparator
    {
       public int compare(Object o1, Object o2)
       {
	   String sz1 = (String) o1;
	   String sz2 = (String) o2;
	   int ntype1;
	   int ntype2;
	   int nchrom1=0;
	   int nchrom2=0;

	  if (sz1.equals("X"))
	  {
	      //X handle like string
	      ntype1 = TYPE_STRING;
	  }
	  else if (sz1.matches("^\\d+$")) 
          {
	      ntype1 = TYPE_NUMERIC;
	      nchrom1 = Integer.parseInt(sz1);
	  }
	  else if (sz1.matches("^[iIvVxXlL]+$")) 
          {
	      ntype1 = TYPE_ROMAN;
	      nchrom1 = Util.romanToNumeric(sz1);
	  }
	  else
	  {
	      ntype1 = TYPE_STRING;
	  }

	  if (sz2.equals("X"))
	  {
	      //X handle like string
	      ntype2 = TYPE_STRING;
	  }
	  else if (sz2.matches("^\\d+$")) 
          {
	      ntype2 = TYPE_NUMERIC;
	      nchrom2 = Integer.parseInt(sz2);
	  }
	  else if (sz2.matches("^[iIvVxXlL]+$")) 
          {
	      ntype2 = TYPE_ROMAN;
	      nchrom2 = Util.romanToNumeric(sz2);
	  }
	  else
	  {
	      ntype2 = TYPE_STRING;
	  }

          if (ntype1 <ntype2)
          {
	     return -1;
	  }
	  else if (ntype1 > ntype2)
	  {
	      return 1;
	  }
	  else if (ntype1 == TYPE_STRING)
	  {
	      return sz1.compareTo(sz2);
	  }
	  else
	  {
	      if (nchrom1 < nchrom2)
	      {
		 return -1;
	      }
	      else if (nchrom1 > nchrom2)
	      {
	         return 1;
	      }
	      else
	      {
		  return 0;
	      }
	  }
       }
    }

    /**
     * Access method on a comparator
     */
    protected Comparator getComparator(int column) 
    {
       if (column == nromancol)
	    return LEXICALROMAN_COMPARATOR;
       Class columnType = tableModel.getColumnClass(column);
       Comparator comparator = (Comparator) columnComparators.get(columnType);
       if (comparator != null) 
       {
          return comparator;
       }
       return LEXICAL_COMPARATOR;
    }
}