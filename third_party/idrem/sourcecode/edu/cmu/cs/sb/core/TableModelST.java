package edu.cmu.cs.sb.core;

import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableModel;
import javax.swing.event.TableModelListener;
import javax.swing.event.TableModelEvent;
import java.util.*;

/**
 *Class implements a standard table model
 */
public class TableModelST extends AbstractTableModel 
{
    /**
     *An array of the column header names
     */
   String[] columnNames;

    /**
     *A multi-dimensional array of data values
     */
   Object[][] data;

    /**
     *Returns the Class type of column c
     */
   public Class getColumnClass(int c) 
   {
      return getValueAt(0, c).getClass();
   }

   /**
    *Always returns false, no cell is editable under this model
    */
   public boolean isCellEditable(int rowIndex, int ColumnIndex)
   {
      return false;
   }

    /**
     *Initializes the table model with the data and columnNames parameters
     */
    
   public TableModelST(Object[][] data, String[] columnNames)
   {
      this.columnNames = columnNames;
      this.data = data;
   }
    

    /**
     *Initializes the table model with the vector vdata and vcolumnNames parameters
     *converting their contents into arrays.  Note 
     *vdata is a vector, and each element of the vector is another vector of the same size
     */
   public TableModelST(Vector vdata, Vector vcolumnNames)
   {
      data = new Object[vdata.size()][((Vector) vdata.get(0)).size()];
      columnNames = new String[vcolumnNames.size()];
 
      for (int nindex = 0; nindex < columnNames.length; nindex++)
      {
         columnNames[nindex] = (String) vcolumnNames.get(nindex);
      }

      for (int nindex = 0; nindex < data.length; nindex++)
      {
         Vector currVector = (Vector) vdata.get(nindex);
         for (int nindex2 = 0; nindex2 < data[nindex].length; nindex2++)
         {
            data[nindex][nindex2] = currVector.get(nindex2);
         }
      }
   }

    /**
     *Returns the number of columns
     */
   public int getColumnCount() 
   {
       return columnNames.length;
   }

    /**
     *Returns the number of rows
     */
   public int getRowCount() 
   {
      return data.length;
   }

    /**
     *Returns the column name of ncol
     */
   public String getColumnName(int ncol)
   {
      return columnNames[ncol];
   }

    /**
     *Returns the value in the table in row nrow and column ncol
     */
   public Object getValueAt(int nrow, int ncol)
   {
       return data[nrow][ncol];
   }

}