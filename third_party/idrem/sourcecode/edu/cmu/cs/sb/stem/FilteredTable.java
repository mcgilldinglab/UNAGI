package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.core.*;
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
import javax.swing.*;
import javax.swing.table.*;
import javax.swing.event.*;
import java.io.*;
import java.awt.datatransfer.*;


/**
 * Class for the table of genes that have been filtered
 */
public class FilteredTable extends JPanel implements ActionListener 
{

    STEM_DataSet theDataSet;


    final static Color bg = Color.white;
    final static Color fg = Color.black;

    String[] columnNames;
    String[][] tabledata;
    JButton saveButton; 
    JButton savenamesButton;
    JButton copyButton;
    JButton copynamesButton;
    JFrame theFrame;
    TableSorter sorter;

    /**
     * Constructor that builds the table
     */
    public FilteredTable(JFrame theFrame,STEM_DataSet theDataSet) 
    {
	this.theFrame = theFrame;
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

        setBackground(bg);
        setForeground(fg);

        this.theDataSet = theDataSet;
  
        columnNames = new String[3];
        columnNames[0] = "Selected";
        columnNames[1] = theDataSet.szGeneHeader;
        columnNames[2] = theDataSet.szProbeHeader;

        tabledata = new String[theDataSet.htFiltered.size()][columnNames.length];
    
        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);

        Iterator geneSet = theDataSet.htFiltered.keySet().iterator();
        int nindex = 0;
        while (geneSet.hasNext())
	{
	    String szgene = (String) geneSet.next();
            String szprobe  = (String) theDataSet.htFiltered.get(szgene);
		   
            tabledata[nindex][1] = szgene;
            tabledata[nindex][2] = szprobe;

            if (theDataSet.tga.isOrder(szgene))
	    {
                tabledata[nindex][0] = "Yes";
            }
            else
	    {
                tabledata[nindex][0] = "";
	    }
	    nindex++;
	}

        sorter = new TableSorter(new TableModelST(tabledata, columnNames));

        final JTable table = new JTable(sorter);
        sorter.setTableHeader(table.getTableHeader());
        table.setPreferredScrollableViewportSize(new Dimension(800,  
                                 Math.min((table.getRowHeight()+table.getRowMargin())*
					  table.getRowCount(),400)));

           TableColumn column;
           column = table.getColumnModel().getColumn(0);
           column.setPreferredWidth(100);
           column = table.getColumnModel().getColumn(1);
           column.setPreferredWidth(350);
           column = table.getColumnModel().getColumn(2);
           column.setPreferredWidth(300);


        //Create the scroll pane and add the table to it.
        JScrollPane scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);

        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);

        copyButton = new JButton("Copy Table",
                                Util.createImageIcon("Copy16.gif"));
        copyButton.setActionCommand("copy");
        copyButton.setMinimumSize(new Dimension(800,20));
        copyButton.addActionListener(this);


        saveButton = new JButton("Save Table", Util.createImageIcon("Save16.gif"));
        saveButton.setActionCommand("save");
        saveButton.setMinimumSize(new Dimension(800,20));
        saveButton.addActionListener(this);
        JPanel buttonPanel = new JPanel();
	buttonPanel.add(copyButton);
        buttonPanel.add(saveButton);
        buttonPanel.setBackground(Color.white);

        copynamesButton = new JButton("Copy Gene Names",
                                Util.createImageIcon("Copy16.gif"));
        copynamesButton.setActionCommand("copynames");
        copynamesButton.setMinimumSize(new Dimension(800,20));
        copynamesButton.addActionListener(this);

        savenamesButton = new JButton("Save Gene Names",
                                Util.createImageIcon("Save16.gif"));
        savenamesButton.setActionCommand("savenames");
        savenamesButton.setMinimumSize(new Dimension(800,20));
        savenamesButton.addActionListener(this);

	buttonPanel.add(copynamesButton);
        buttonPanel.add(savenamesButton);

        JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpButton.addActionListener(this);
        helpButton.setActionCommand("help");
        buttonPanel.add(helpButton);
	buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel);
    }


    /**
     * Handles response to buttons being pressed
     */
    public void actionPerformed(ActionEvent e)
    {
	String szCommand = e.getActionCommand();
       if (szCommand.equals("copy"))
       {
	   writeToClipboard();
       }
       else if (szCommand.equals("copynames"))
       {
	   writenamesToClipboard();
       }
       else if ((szCommand.equals("save"))||(szCommand.equals("savenames")))
       {
          try
          {
             int nreturnVal = Util.theChooser.showSaveDialog(this);
             if (nreturnVal == JFileChooser.APPROVE_OPTION) 
             {
                File f = Util.theChooser.getSelectedFile();
                PrintWriter pw = new PrintWriter(new FileOutputStream(f));
                if (szCommand.equals("save"))
		{
                   printFile(pw);
		}
                else
		{
		    printGeneList(pw);
		}
                pw.close();
	     }
          }
          catch (FileNotFoundException ex)
          {
              final FileNotFoundException fex = ex;
              javax.swing.SwingUtilities.invokeLater(new Runnable() 
              {
                 public void run() 
                 {
                    JOptionPane.showMessageDialog(null, fex.getMessage(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);
	         }
	      });
              ex.printStackTrace(System.out);
          }
       }
       else if (szCommand.equals("help"))
       {
           String szMessage;

	   if (theDataSet.bkmeans)
	   {
              szMessage = "The table contains genes included in the data set file, but that did not "
                           +"pass filter according to filtering criteria specified under the advanced "
                           +"options. \n\n"
                           +"The table has three columns: \n"
                           +"*  Selected - contains a 'Yes' if the gene is part the GO Category or Query "
                           +"Set that the K-means clusters "+
                           "are currently ordered by.\n"
                          +"*  "+theDataSet.szGeneHeader+" - The name of the gene.\n"
                          +"*  "+theDataSet.szProbeHeader+" - The spot ID(s) associated with the gene.\n\n"
                          +"Note:\n"
                          +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
                          +"+Using the 'Copy Table' the entire table can be copied to the clipboard, or "+
                           "with the 'Copy Gene Names' button just the gene names.\n"
		           +"+Using the 'Save Table' button the entire table can be saved, or using the "+
			  "'Save Gene Names' button just the gene names.";
	   }
	   else
	   {
              szMessage = "The table contains genes included in the data set file, but that did not "
                           +"pass filter according to filtering criteria specified under the advanced "
                           +"options. \n\n"
                           +"The table has three columns: \n"
                           +"*  Selected - contains a 'Yes' if the gene is part the GO Category or Query "
                           +"Set that the profiles or cluster of profiles "+
                           "are currently ordered by, otherwise is blank.\n"
                          +"*  "+theDataSet.szGeneHeader+" - The name of the gene.\n"
                          +"*  "+theDataSet.szProbeHeader+" - The spot ID(s) associated with the gene.\n\n"
                          +"Note:\n"
                          +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
                          +"+Using the 'Copy Table' the entire table can be copied to the clipboard, or "+
                           "with the 'Copy Gene Names' button just the gene names.\n"
		           +"+Using the 'Save Table' button the entire table can be saved, or using the "+
			  "'Save Gene Names' button just the gene names.";
	   }

	   Util.renderDialog(theFrame,szMessage);
       }
    }

    /**
     * Prints the genes in the table to a file
     */
    public void printGeneList(PrintWriter pw)
    {
       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
	  pw.println(sorter.getValueAt(nrow,1));
       }
    }

    /**
     * Copies the names in the table to the clipboard
     */
    public void writenamesToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();

       	// get the system clipboard
       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
	   sbuf.append(sorter.getValueAt(nrow,1)+"\n");
       }

       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }

    /**
     * Prints the contents of the table in the file
     */
    public void printFile(PrintWriter pw)
    {
        pw.println("Table of Genes Filtered");

       for (int ncol = 0; ncol < columnNames.length-1; ncol++)
       {
          pw.print(columnNames[ncol] +"\t");
       }
       pw.println(columnNames[columnNames.length - 1]);

       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
          for (int ncol = 0; ncol < tabledata[nrow].length-1; ncol++)
          {
	      pw.print(sorter.getValueAt(nrow,ncol)+"\t");
          }
          pw.println(sorter.getValueAt(nrow,columnNames.length-1));
       }
    }

    /**
     * Copies the contents of the table to a clipboard
     */
    public void writeToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();

       for (int ncol = 0; ncol < columnNames.length-1; ncol++)
       {
          sbuf.append(columnNames[ncol] +"\t");
       }
       sbuf.append(columnNames[columnNames.length - 1]+"\n");

       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
          for (int ncol = 0; ncol < tabledata[nrow].length-1; ncol++)
          {
	      sbuf.append(sorter.getValueAt(nrow,ncol)+"\t");
          }
          sbuf.append(sorter.getValueAt(nrow,columnNames.length-1)+"\n");
       }

       	// get the system clipboard
       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }
}

