package edu.cmu.cs.sb.core;

import java.io.*;
import java.util.*;
import java.awt.*;
import javax.swing.*;
import javax.swing.table.*;
import javax.swing.border.*;


/**
 *Class implements multiple lines in a table header
 *Code based on
 *http://www.codeguru.com/java/articles/126.shtml
 */
public class MultiLineHeaderRenderer extends JList implements TableCellRenderer 
{


   TableSorter theTableSorter;

    /**
     *Initializes the TableSorter object to the header.  
     *Sets opaque to true
     *sets Foreground to UIManager.getColor("TableHeader.foreground")
     *sets Background to UIManager.getColor("TableHeader.background")
     *if not on a mac, sets Border to UIManager.getBorder("TableHeader.cellBorder")
     *sets cellRenderer
     */
   public MultiLineHeaderRenderer(TableSorter theTableSorter) 
   {
      this.theTableSorter = theTableSorter;
      setOpaque(true);

      setForeground(UIManager.getColor("TableHeader.foreground"));
      setBackground(UIManager.getColor("TableHeader.background"));
      if (!System.getProperty("os.name").toLowerCase(Locale.ENGLISH).startsWith("mac"))
      {
         //mac's do not like this line
         setBorder(UIManager.getBorder("TableHeader.cellBorder"));
      }
      
      ListCellRenderer renderer = getCellRenderer();
      ((JLabel)renderer).setHorizontalAlignment(JLabel.CENTER);
      setCellRenderer(renderer);
   }


    /**
     *Class extends JLabel implements ListCellRenderer
     */
   class MyCellRenderer extends JLabel implements ListCellRenderer 
   {
      // This is the only method defined by ListCellRenderer.
      // We just reconfigure the JLabel each time we're called.
      public MyCellRenderer(JTable table, int column)
      {
         this.column = column;
         this.table  = table;
      }

      JTable table;
      int column;

      public Component getListCellRendererComponent(
         JList list,
         Object value,            // value to display
         int index,               // cell index
         boolean isSelected,      // is the cell selected
         boolean cellHasFocus)    // the list and the cell have the focus
      {
         String s = value.toString();
         setText(s);
         setHorizontalAlignment(JLabel.CENTER);
	 setHorizontalTextPosition(JLabel.LEFT);
         int modelColumn = table.convertColumnIndexToModel(column);
	 
         if (index == 0)
	 {
            setIcon(theTableSorter.getHeaderRendererIcon(modelColumn, getFont().getSize()));
	 }
         else
	 {
	     setIcon(null);
	 }
	
	 setFont(list.getFont());
         setOpaque(true);
	 
         return this;
      }
   }
 
    /**
     * Renders the data in value
     */
   public Component getTableCellRendererComponent(JTable table, Object value,
		       boolean isSelected, boolean hasFocus, int row, int column) 
   {
      setFont(table.getFont());
      String str = (value == null) ? "" : value.toString();
      BufferedReader br = new BufferedReader(new StringReader(str));
      String line;

      Vector v = new Vector();
      try 
      {
         while ((line = br.readLine()) != null) 
         {
            v.addElement(line);	
	 }
      } 
      catch (IOException ex) 
      {
         ex.printStackTrace();
      }
         
      setListData(v);
      setCellRenderer(new MyCellRenderer(table,column));

      return this;
   }
    
}

