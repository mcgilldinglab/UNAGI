package edu.cmu.cs.sb.core;

import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableModel;
import javax.swing.event.TableModelListener;
import javax.swing.event.TableModelEvent;
import java.util.*;

/**
 *Class implements a table model, which is used for the define gene set table
 */
public class TableModelSetST extends AbstractTableModel 
{
    /**
     *An array of the column header names
     */
   String[] columnNames;

    /**
     *A multi-dimensional array of data values
     */
   public Object[][] data;

    /**
     *The GoAnnotations object to update based on which genes are selected
     */
   GoAnnotations tga;

    /**
     *Returns the Class type of column c
     */
   public Class getColumnClass(int c) 
   {
      return getValueAt(0, c).getClass();
   }

   /**
    *Returns true if ColumnIndex is 1, otherwise returns false
    */
   public boolean isCellEditable(int rowIndex, int ColumnIndex)
   {
      return (ColumnIndex == 1);
   }

    /**
     *Initializes the table model with the data, columnNames, and tga parameters
     */
   public TableModelSetST(Object[][] data, String[] columnNames,GoAnnotations tga)
   {
      this.columnNames = columnNames;
      this.data = data;
      this.tga = tga;
   }

    /**
     *Initializes the table model with the vector vdata and vcolumnNames parameters
     *converting their contents into arrays.  Note 
     *vdata is a vector, and each element of the vector is another vector of the same size
     */
   public TableModelSetST(Vector vdata, Vector vcolumnNames,GoAnnotations tga)
   {
      data = new Object[vdata.size()][((Vector) vdata.get(0)).size()];
      columnNames = new String[vcolumnNames.size()];
      this.tga = tga;
 
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

    /**
     *Sets the value in the table in row nrow and column ncol
     *to the contents of value and then calls fireTableCellUpdated.
     *This method will only be called with ncol set to 1 since that is
     *the only editable table in the column.
     *value is assumed to be a Boolean object and thus true or false.
     *Updates the tga.nGeneSet count, to the number of genes selected.
     *Updates whether the gene in the row is selected in the tga.htGeneNames table.
     */
   public void setValueAt(Object value, int row, int col) 
   {
      if ((((Boolean) value).booleanValue()) && 
         (!((Boolean) data[row][col]).booleanValue()))
      {
	  //selecting a previously unselected row
         tga.nGeneSet++;
      }
      else if (!(((Boolean) value).booleanValue()) && 
		 (((Boolean) data[row][col]).booleanValue()))
      {
	  //unselecting a previously unselected row
         tga.nGeneSet--;
      }
      //updating tga.htGeneNames
      tga.htGeneNames.put(data[row][0], value);

      data[row][col] = value;
      fireTableCellUpdated(row, col);
   }
}