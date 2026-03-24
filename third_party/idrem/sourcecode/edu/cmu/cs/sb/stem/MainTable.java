package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.core.*;
import edu.cmu.cs.sb.chromviewer.*;

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
 * Class which contains table for all genes that were not filtered.
 */
public class MainTable extends JPanel implements ActionListener 
{
    private boolean DEBUG = false;

    STEM_DataSet theDataSet;  
    ChromFrame cf;
    final static Color bg = Color.white;
    final static Color fg = Color.black;

    String[] columnNames;
    String[][] tabledata;
    JButton chromButton;
    JButton saveButton;
    JButton savenamesButton;
    JButton copyButton;
    JButton copynamesButton;
    JFrame theFrame;
    TableSorter sorter;

    /**
     * Constructor to build a new main table
     */
    public MainTable(JFrame theFrame,STEM_DataSet theDataSet, final GenePlotPanel thegeneplotpanel,final ChromFrame cf) 
    {
        this.theFrame = theFrame;
	this.cf = cf;
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

        setBackground(bg);
        setForeground(fg);

        this.theDataSet = theDataSet;
  
        columnNames = new String[theDataSet.numcols+4];
        columnNames[0] = "Selected";
        columnNames[1] = theDataSet.szGeneHeader;
        columnNames[2] = theDataSet.szProbeHeader;
	if (theDataSet.bkmeans)
	{
           columnNames[3] = "Cluster";
	}
	else
	{
           columnNames[3] = "Profile";
	}

        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
	{
            columnNames[ncolindex+4] = ""+theDataSet.dsamplemins[ncolindex];
	}

        tabledata = new String[theDataSet.numrows][columnNames.length];
    
        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);

        for (int nindex = 0; nindex < theDataSet.numrows; nindex++)
	{		   
            tabledata[nindex][1] = theDataSet.genenames[nindex];
            tabledata[nindex][2] = theDataSet.probenames[nindex];

            if (theDataSet.tga.isOrder(tabledata[nindex][1]))
	    {
                tabledata[nindex][0] = "Yes";
            }
            else
	    {
                tabledata[nindex][0] = "";
	    }

            ArrayList bestAssignments = theDataSet.bestassignments[nindex];
            StringBuffer szassignedbuf = new StringBuffer(""+((Integer) bestAssignments.get(0)).intValue());

            for (int njindex = 1; njindex < bestAssignments.size(); njindex++)
	    {
		szassignedbuf.append(";"+((Integer) bestAssignments.get(njindex)).intValue());
            }

	    String szassigned = szassignedbuf.toString();
            tabledata[nindex][3] = szassigned;

            for (int ncol = 0; ncol < theDataSet.numcols; ncol++)
	    {
               if (theDataSet.pmavalues[nindex][ncol]==0)
	       {
	          tabledata[nindex][ncol+4] = "";
               }
               else
	       {
                 tabledata[nindex][ncol+4]= nf2.format(theDataSet.data[nindex][ncol]);
	       }
	    }	      
	}

        sorter = new TableSorter(new TableModelST(tabledata, columnNames));

        final JTable table = new JTable(sorter);
        sorter.setTableHeader(table.getTableHeader());
        table.setPreferredScrollableViewportSize(new Dimension(800,  
                               Math.min((table.getRowHeight()+table.getRowMargin())*
					table.getRowCount(),400)));

        TableColumn column;
        column = table.getColumnModel().getColumn(0);
        column.setPreferredWidth(50);
        column = table.getColumnModel().getColumn(1);
        column.setPreferredWidth(200);
        column = table.getColumnModel().getColumn(2);
        column.setPreferredWidth(150);
        column = table.getColumnModel().getColumn(3);
        column.setPreferredWidth(45);


        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
        {
           column = table.getColumnModel().getColumn(ncolindex+4);
           column.setPreferredWidth(45);
        }        
        //Create the scroll pane and add the table to it.
        JScrollPane scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);


        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        ListSelectionModel rowSM = table.getSelectionModel();
        final STEM_DataSet finaltheDataSet = theDataSet;
        final TableSorter finalsorter = sorter;
        rowSM.addListSelectionListener(new ListSelectionListener() 
        {
           public void valueChanged(ListSelectionEvent e) 
           {
                 //Ignore extra messages.
                 if (e.getValueIsAdjusting()) return;

                 final ListSelectionModel lsm = (ListSelectionModel)e.getSource();
                 if (lsm.isSelectionEmpty()) 
                 {
                    return;
                 } 
                 else 
                 {
		    javax.swing.SwingUtilities.invokeLater(new Runnable() 
                    {
                       public void run() 
                       {

                          int nselectedRow = lsm.getMinSelectionIndex();
                          String szprofile = (String) finalsorter.getValueAt(nselectedRow,3); 
                          String szgene = (String) finalsorter.getValueAt(nselectedRow,1);
                          String szspot = (String) finalsorter.getValueAt(nselectedRow,2);
                          StringTokenizer stprofile = new StringTokenizer(szprofile,";");
                          while (stprofile.hasMoreTokens())
		          {
			     int nprofile = Integer.parseInt(stprofile.nextToken());
                    
                             ProfileGui pg = 
				 new ProfileGui(finaltheDataSet,nprofile,null,
						null,-1,null,szgene,szspot, thegeneplotpanel,cf);        

			       pg.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
			       pg.pack();
			       pg.setSize(new Dimension(MAINGUI2.SCREENWIDTH,MAINGUI2.SCREENHEIGHT));
			       pg.setVisible(true);
	  	          }
  		        }
		     });
		 }
	   }
        });

        saveButton = new JButton("Save Table", Util.createImageIcon("Save16.gif"));
        saveButton.setActionCommand("save");
        saveButton.setMinimumSize(new Dimension(800,20));
        saveButton.addActionListener(this);


        copyButton = new JButton("Copy Table",  Util.createImageIcon("Copy16.gif"));
        copyButton.setActionCommand("copy");
        copyButton.setMinimumSize(new Dimension(800,20));
        copyButton.addActionListener(this);

        chromButton = new JButton("Chromosome View");
	if (cf.genomeParser.szchromval.equals(""))
	{
	    chromButton.setEnabled(false);
	}
	else
	{
	    chromButton.setEnabled(true);
	}
        chromButton.setActionCommand("chromview");
        chromButton.setMinimumSize(new Dimension(800,20));
        chromButton.addActionListener(this);

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
	buttonPanel.add(chromButton);
        buttonPanel.add(helpButton);
	buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel);
    }

    /**
     * Responds to buttons being pressed
     */
    public void actionPerformed(ActionEvent e)
    {
       String szCommand = e.getActionCommand();

       if (szCommand.equals("chromview"))
       {
	   chromView();
       }
       else if (szCommand.equals("copy"))
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
           String szlogword ="";
	   if (theDataSet.btakelog)
	   {
	      szlogword ="log base two ";
	   }
           String szMessage;

	   if (theDataSet.bkmeans)
	   {
              szMessage = "The table contains the list of genes included in the data set file and "
                           +"passed filter according to filering criteria specified under the advanced "
                           +"options.\n\n"
                           +"The first four columns are: \n"
                           +"*Selected - contains a 'Yes' if the gene is part the GO Category or Query "
                           +"Set that the K-means clusters "+
                           "are currently ordered by, otherwise is blank.\n"
                           +"*  "+theDataSet.szGeneHeader+" - The name of the gene.\n"
                           +"*  "+theDataSet.szProbeHeader+" - The spot ID(s) associated with the gene.\n"
                           +"*Cluster - The cluster(s) the gene belongs to.\n\n"
                           +"The remaining columns contain the "+szlogword+
                           "expression change relative to the first time point.\n\n"+
                           "Note:\n"
                           +"+Clicking on a row display the profile(s) to which the gene was assigned.\n"
                           +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
                           +"Using the 'Copy Table' the entire table can be copied to the clipboard, or "+
                           "with the 'Copy Gene Names' button just the gene names.\n"
		           +"Using the 'Save Table' button the entire table can be saved, or using the "+
			   "'Save Gene Names' button just the gene names.\n"+
                          "'Chromosome View' button plots the genes in the table on the Chromosome Viewer and makes "+
                          "the viewer visible if not already so. If the button is inactive then no chromosome locations "+
		          "were specified.";
	   }
	   else
	   {
              szMessage = "The table contains the list of genes included in the data set file and "
                           +"passed filter according to filering criteria specified under the advanced "
                           +"options.\n\n"
                           +"The first four columns are: \n"
                           +"*Selected - contains a 'Yes' if the gene is part the GO Category or Query "
                           +"Set that the profiles or cluster of profiles "+
                           "are currently ordered by, otherwise is blank.\n"
                           +"*  "+theDataSet.szGeneHeader+" - The name of the gene.\n"
                           +"*  "+theDataSet.szProbeHeader+" - The spot ID(s) associated with the gene.\n"
                           +"*Profile - The profile the gene was assigned to, in case of a tie "+ 
	                   "multiple profiles are listed.\n\n"
                           +"The remaining columns contain the "+szlogword+
                           "expression change relative to the first time point.\n\n"+
                           "Note:\n"
                           +"+Clicking on a row display the profile(s) to which the gene was assigned.\n"
                           +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
                           +"Using the 'Copy Table' the entire table can be copied to the clipboard, or "+
                           "with the 'Copy Gene Names' button just the gene names.\n"
		           +"Using the 'Save Table' button the entire table can be saved, or using the "+
			   "'Save Gene Names' button just the gene names.\n"+
                           "+'Chromosome View' button plots the genes in the table on the Chromosome Viewer and makes "+
                           "the viewer visible if not already so. If the button is inactive then no chromosome locations "+
		           "were specified.";

	   }

	   Util.renderDialog(theFrame,szMessage);
       }
    }

    /**
     * Calls the chromosome viewer to draw and display gene on this table
     */
    private void chromView() 
    {	
       String[] chromgeneNames = new String[tabledata.length];
       for(int i = 0; i < chromgeneNames.length; i++)
       {
	  chromgeneNames[i] = tabledata[i][1];
       } 
       cf.drawGenes(chromgeneNames);
       cf.setVisible(true);    
    }
   

    /**
     * Writes the genes in the table to a file
     */
    public void printGeneList(PrintWriter pw)
    {
       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
	  pw.println(sorter.getValueAt(nrow,1));
       }
    }

    /**
     * Copies the gene names in the file to a clipboard
     */
    public void writenamesToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();

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
     * Writes the content of the table to a file
     */
    public void printFile(PrintWriter pw)
    {
        pw.println("Table of Genes Passing Filter");

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
     * Copies the content of the table to the clipboard
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

