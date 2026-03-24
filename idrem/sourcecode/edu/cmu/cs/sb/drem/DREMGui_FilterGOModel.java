package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import javax.swing.table.AbstractTableModel;
import javax.swing.table.TableModel;
import javax.swing.event.TableModelListener;
import javax.swing.event.TableModelEvent;
import java.util.*;

/**
 *Class implements a table model, which is used for the define gene set table
 */
public class DREMGui_FilterGOModel extends AbstractTableModel {
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
	public Class getColumnClass(int c) {
		return getValueAt(0, c).getClass();
	}

	/**
	 *Returns true if ColumnIndex is 1, otherwise returns false
	 */
	public boolean isCellEditable(int rowIndex, int ColumnIndex) {
		return false;
	}

	/**
	 *Initializes the table model with the data, columnNames, and tga
	 * parameters
	 */
	public DREMGui_FilterGOModel(Object[][] data, String[] columnNames) {
		this.columnNames = columnNames;
		this.data = data;
	}

	/**
	 *Returns the number of columns
	 */
	public int getColumnCount() {
		return columnNames.length;
	}

	/**
	 *Returns the number of rows
	 */
	public int getRowCount() {
		return data.length;
	}

	/**
	 *Returns the column name of ncol
	 */
	public String getColumnName(int ncol) {
		return columnNames[ncol];
	}

	/**
	 *Returns the value in the table in row nrow and column ncol
	 */
	public Object getValueAt(int nrow, int ncol) {
		return data[nrow][ncol];
	}

	/**
	 *Sets the value in the table in row nrow and column ncol to the contents
	 * of value and then calls fireTableCellUpdated. This method will only be
	 * called with ncol set to 1 since that is the only editable table in the
	 * column. value is assumed to be a Boolean object and thus true or false.
	 * Updates the tga.nGeneSet count, to the number of genes selected. Updates
	 * whether the gene in the row is selected in the tga.htGeneNames table.
	 */
	public void setValueAt(Object value, int row, int col) {
		data[row][col] = value;
		fireTableCellUpdated(row, col);
	}
}