package edu.cmu.cs.sb.core;

//import edu.cmu.cs.sb.core.*;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.*;
import java.text.NumberFormat;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.table.*;
import java.util.zip.*;

/**
 *Class encapsulates a panel which is used to display the direct contents of a file in table format
 */
public class DataTable extends JPanel 
{

    String szfile;
    /**
     *Constructor for the class
     *szfile is the file containing the data to be displayed
     *If bhorizontal is true then table resize mode is turned off
     */
    public DataTable(JFrame stframe, String szfile, boolean bhorizontal)
    {
	final JFrame fframe = stframe;

        BufferedReader br = null;

        try
        {
	   this.szfile = szfile;

           //tries first reading as a GZIP then a regular
           try
           {
              br = new BufferedReader(new InputStreamReader(
                             new GZIPInputStream(new FileInputStream(szfile))));
 	   }
           catch (IOException ex)
           {
              br = new BufferedReader(new FileReader(szfile));
           }


	   String szLine = br.readLine();
	   int ncols;
	   Vector vcolumnNames;
	   StringTokenizer st;
	   StringTokenizer st2;
	   if (szLine == null)
	   {
	      ncols = 0;
	      vcolumnNames = new Vector();
	   }
	   else
	   {
	       //reads in header line
	      st = new StringTokenizer(szLine,"\t");
	      st2 = new StringTokenizer(szLine,"\t",true);
              ncols = st2.countTokens() - st.countTokens()+1;
	      vcolumnNames = new Vector(ncols);

              for (int nindex = 0; nindex <ncols; nindex++)
	      {
	         if (!st2.hasMoreTokens())
	         {
                    vcolumnNames.add("");
	         }
	         else
	         {
	            String sztoken = st2.nextToken();
	            if (!sztoken.equals("\t"))
	            {
                       vcolumnNames.add(sztoken);
		       if (st2.hasMoreTokens())
		       {
                          st2.nextToken();
		       }
		    }
	            else
	            {
                       vcolumnNames.add("");
	            }
	         }
	      }
	   }

	   //reads in the rest of the data
  	   Vector vtabledata = new Vector();
	   while ((szLine = br.readLine())!=null)
	   {
	      st2 = new StringTokenizer(szLine,"\t",true);
	      Vector vrow = new Vector(ncols);
              for (int nindex = 0; nindex <ncols; nindex++)
	      {
                 if (st2.hasMoreTokens())
     	         {
	            String sztoken = st2.nextToken();
	            if (!sztoken.equals("\t"))
	            {
                       vrow.add(sztoken);
		       if (st2.hasMoreTokens())
                          st2.nextToken();
	            }
      	            else
	            {
                       vrow.add("");
	            }
		 }
		 else
		 {
                    vrow.add("");
		 }
	      }
 	      vtabledata.add(vrow);
	   }
	   
           setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
           TableSorter sorter = new TableSorter(new TableModelST(vtabledata,vcolumnNames));
           final JTable table = new JTable(sorter);
	   if (bhorizontal)
	   {
  	      table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
	   }
           sorter.setTableHeader(table.getTableHeader());
           table.setPreferredScrollableViewportSize(new Dimension(800, 
                             Math.min((table.getRowHeight()+table.getRowMargin())*
				      table.getRowCount(),400)));
           JScrollPane scrollPane = new JScrollPane(table);
           add(scrollPane);
        }
	catch (FileNotFoundException ex)
	{
           final FileNotFoundException fex = ex;

           javax.swing.SwingUtilities.invokeLater(new Runnable() 
           {
              public void run() 
              {
                 JOptionPane.showMessageDialog(fframe, fex.getMessage(), 
                                  "Error", JOptionPane.ERROR_MESSAGE);
                 fex.printStackTrace(System.out);
	       }
	    });
	}
        catch (IOException ex)
        {
           final IOException fex = ex;

           javax.swing.SwingUtilities.invokeLater(new Runnable() 
           {
              public void run() 
              {
                 JOptionPane.showMessageDialog(fframe, fex.getMessage(), 
                                  "Error", JOptionPane.ERROR_MESSAGE);
                 fex.printStackTrace(System.out);
	       }
	    });
	 }
 	 finally
	 {
	     try
	     {
		 if (br != null)
		 {
		     br.close();
		 }
	     }
	     catch(IOException ex)
	     {
                final IOException fex = ex;

                javax.swing.SwingUtilities.invokeLater(new Runnable() 
                {
                   public void run() 
                   { 
                      JOptionPane.showMessageDialog(fframe, fex.getMessage(), 
                                  "Error", JOptionPane.ERROR_MESSAGE);
                      fex.printStackTrace(System.out);
	           }
	        });
	     }
	 }
    }
}