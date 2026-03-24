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
import java.io.*;
import javax.swing.*;
import javax.swing.table.*;
import java.awt.datatransfer.*;

/**
 * Class for a table of genes assigned to a profile or cluster
 */
public class GeneTable extends JPanel implements ActionListener 
{
    private boolean DEBUG = false;

    ChromFrame cf;
    STEM_DataSet theDataSet;
    int nprofile; 
    String szAssignInfo;
    String[] columnNames;
    String[][] tabledata;
    JButton saveButton;
    JButton copyButton;
    JButton savenamesButton;
    JButton copynamesButton;
    JButton chromButton;
    ArrayList alprofiles;
    TableSorter sorter;
    final static Color bg = Color.white;
    final static Color fg = Color.black;

    boolean bprofileonly;
    boolean bquery;
    String szTitle;
    JFrame theFrame;
    HashSet htinames;


    /**
     * Constructor to display genes associated with a single profile or cluster
     */
    public GeneTable(JFrame theFrame, STEM_DataSet theDataSet, int nprofile, String szAssignInfo,
                     boolean bquery,HashSet htinames,String szTitle,ChromFrame cf)
    {
        this.theFrame = theFrame;
	this.nprofile = nprofile;
        alprofiles = new ArrayList();
        alprofiles.add(new Integer(nprofile));
	bprofileonly = true;
        this.szTitle = szTitle;
	this.cf = cf;
  
        geneTableHelper(theDataSet,  szAssignInfo,bquery,htinames);
    }

    /**
     * Constructor to display genes associated with a set of profiles
     */
    public GeneTable(JFrame theFrame, STEM_DataSet theDataSet, ArrayList alprofilerec, String szAssignInfo,
		     boolean bquery, HashSet htinames,String szTitle,ChromFrame cf)
    {
        this.theFrame = theFrame;
	this.cf = cf;
        int numrecs = alprofilerec.size();
        alprofiles = new ArrayList();
        for (int nindex = 0; nindex < numrecs; nindex++)
	{
            STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) alprofilerec.get(nindex);
            alprofiles.add(new Integer(pr.nprofileindex));
	}
        bprofileonly = false;
        this.szTitle = szTitle;
        geneTableHelper(theDataSet, szAssignInfo,bquery,htinames);
    }


    /**
     * Builds the gene table interface
     */
    public void geneTableHelper(STEM_DataSet theDataSet, String szAssignInfo,
                                boolean bquery,HashSet htinames) 
    {

        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

        setBackground(bg);
        setForeground(fg);
        this.bquery = bquery;
        this.htinames = htinames;
        this.theDataSet = theDataSet;
        this.szAssignInfo = szAssignInfo;  

        int numcols = theDataSet.numcols + 4;
        columnNames = new String[numcols];
        columnNames[0] = "Selected";
        columnNames[1] = "Weight";
        columnNames[2] = theDataSet.szGeneHeader;
        columnNames[3] = theDataSet.szProbeHeader;


        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
	{
            columnNames[ncolindex+4] = ""+theDataSet.dsamplemins[ncolindex];
	}
 
        int ngeneindex = 0;

        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);

        int ntabledatasize = 0;
        if (htinames != null)
	{
	   ntabledatasize = htinames.size();
        } 
        else
	{
           for (int nalindex = 0; nalindex < alprofiles.size(); nalindex++)
	   {
              int nprofileindex = ((Integer) alprofiles.get(nalindex)).intValue();

              if (bquery)
	      {
		  //query table
                 ArrayList profileBestIndex = theDataSet.profilesAssigned[nprofileindex];
                 int npbs = profileBestIndex.size();
                 for (int nindex = 0; nindex < npbs; nindex++)
	         {
 	            ngeneindex = ((Integer) profileBestIndex.get(nindex)).intValue();
                    if (theDataSet.tga.isOrder(theDataSet.genenames[ngeneindex]))
	            {
		       ntabledatasize++;
	            }
	         }
	      } 
              else
	      {
	         ntabledatasize += theDataSet.profilesAssigned[nprofileindex].size();
	      }
	   }
	}

 
        tabledata = new String[ntabledatasize][columnNames.length];
       
        int ntableindex = 0;
        for (int nalindex = 0; nalindex < alprofiles.size(); nalindex++)
	{
	   int nprofileindex = ((Integer)  alprofiles.get(nalindex)).intValue();
           ArrayList profileBestIndex = theDataSet.profilesAssigned[nprofileindex];
           int npbs = profileBestIndex.size();
           for (int nindex = 0; nindex < npbs; nindex++)
	   {
 	      ngeneindex = ((Integer) profileBestIndex.get(nindex)).intValue();

	      if (((!bquery)&&(htinames == null))|| 
                  ((htinames != null) && (htinames.contains(theDataSet.genenames[ngeneindex])))||
		  ((htinames == null) &&
                    (theDataSet.tga.isOrder(theDataSet.genenames[ngeneindex]))))
	      {
		 tabledata[ntableindex][2] = theDataSet.genenames[ngeneindex];  
                 tabledata[ntableindex][3] = theDataSet.probenames[ngeneindex];

                 if ((bquery)|| 
		     (theDataSet.tga.isOrder(tabledata[ntableindex][2])))
	         {
                    tabledata[ntableindex][0] = "Yes";
                 }
                 else
	         {
                    tabledata[ntableindex][0] = "";
	         }

                 tabledata[ntableindex][1] = 
                    nf2.format(1.0/(double)theDataSet.bestassignments[ngeneindex].size());

                 for (int ncol = 0; ncol < theDataSet.numcols; ncol++)
	         {
                    if (theDataSet.pmavalues[ngeneindex][ncol]==0)
	            {
	               tabledata[ntableindex][ncol+4] = "";
                    }
                    else
	            {
                       tabledata[ntableindex][ncol+4]= nf2.format(theDataSet.data[ngeneindex][ncol]);
	            }
		 }
                 ntableindex++;
	      }
	   }
	}

        sorter = new TableSorter(new TableModelST(tabledata,columnNames));
        final JTable table = new JTable(sorter);

        sorter.setTableHeader(table.getTableHeader());
        table.setPreferredScrollableViewportSize(new Dimension(800, 
                             Math.min((table.getRowHeight()+table.getRowMargin())*
				      table.getRowCount(),400)));

        TableColumn column;
        column = table.getColumnModel().getColumn(0);
        column.setPreferredWidth(40);
        column = table.getColumnModel().getColumn(1);
        column.setPreferredWidth(40);
        column = table.getColumnModel().getColumn(2);
        column.setPreferredWidth(200);
        column = table.getColumnModel().getColumn(3);
        column.setPreferredWidth(150);
        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
	{
            column = table.getColumnModel().getColumn(ncolindex+4);
            column.setPreferredWidth(45);
	}

        //Create the scroll pane and add the table to it.
        JScrollPane scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);
        saveButton = new JButton("Save Table",
                                Util.createImageIcon("Save16.gif"));
        saveButton.setActionCommand("save");
        saveButton.setMinimumSize(new Dimension(800,20));
        saveButton.addActionListener(this);

        copyButton = new JButton("Copy Table",
                                Util.createImageIcon("Copy16.gif"));
        copyButton.setActionCommand("copy");
        copyButton.setMinimumSize(new Dimension(800,20));
        copyButton.addActionListener(this);

        
	JPanel buttonPanel =new JPanel();
	buttonPanel.add(copyButton);
        buttonPanel.setBackground(Color.white);
        buttonPanel.add(saveButton);

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
	buttonPanel.add(chromButton);
     
        JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpButton.addActionListener(this);
        helpButton.setActionCommand("help");
        buttonPanel.add(helpButton);
	buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel);
    }

    /**
     * Outputs the content of the PrintWriter
     */
    public void printFile(PrintWriter pw)
    {
       if (bprofileonly)
       {
           pw.print(szTitle+" (");

          for (int nindex = 0; nindex < theDataSet.numcols-1; nindex++)
          {
             pw.print(theDataSet.modelprofiles[nprofile][nindex]+",");
          }
          pw.println(theDataSet.modelprofiles[nprofile][theDataSet.numcols-1]+")");

          pw.println("Num. Genes Expected\tNum. Genes Assigned\tP-Value\tSignificant");
          pw.println(szAssignInfo);
       } 
       else
       {
	   pw.println(szTitle);

           for (int nprofilealindex = 0; nprofilealindex < alprofiles.size(); nprofilealindex++)
           {
	      int nprofileindex = ((Integer) alprofiles.get(nprofilealindex)).intValue();
              pw.print("\t Profile "+nprofileindex+" (");
              for (int nindex = 0; nindex < theDataSet.numcols-1; nindex++)
              {
                 pw.print(theDataSet.modelprofiles[nprofileindex][nindex]+",");
              } 
              pw.println(theDataSet.modelprofiles[nprofileindex][theDataSet.numcols-1]+")");
           } 
       }


       pw.println("-------");

       for (int ncol = 0; ncol < columnNames.length-1; ncol++)
       {
          pw.print(columnNames[ncol] +"\t");
       }
       pw.println(columnNames[columnNames.length - 1]);

       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
          for (int ncol = 0; ncol < tabledata[nrow].length-1; ncol++)
          {
	      pw.print(sorter.getValueAt(nrow,ncol) +"\t");
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
	      sbuf.append(sorter.getValueAt(nrow,ncol) +"\t");
          }
          sbuf.append(sorter.getValueAt(nrow,columnNames.length-1)+"\n");
       }

       	// get the system clipboard
       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }

    /**
     * Copies the gene name to the clipboard
     */
    public void writenamesToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();
       	// get the system clipboard
       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
	  sbuf.append(sorter.getValueAt(nrow,2)+"\n");
       }

       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }

    /**
     * Print the genes in the table to file
     */
    public void printGeneList(PrintWriter pw)
    {
       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
	  pw.println(sorter.getValueAt(nrow,2));
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
	  chromgeneNames[i] = tabledata[i][2];
      }
 
      cf.drawGenes(chromgeneNames);
      cf.setVisible(true);    
    }
    
    /**
     * Responds to button being pressed 
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
           String szMessage = "The table contains genes assigned to the selected ";

           if (bprofileonly)
	   {
	       if (theDataSet.bkmeans)
	       {
		   szMessage += "K-means cluster";
	       }
	       else
	       {
	          szMessage += "profile";
	       }
	   }
           else
	   {
               szMessage += "cluster of profiles (any profile the same color as this one)";
	   }

	   if (htinames != null)
	   {
	       if (theDataSet.bkmeans)
	       {
	          szMessage += " and to the K-means cluster to the immediate left of the yellow "+
                               "column in the comparison display";
	       }
	       else
	       {
	          szMessage += " and to the profile to the immediate left of the yellow "+
                               "column in the comparison display";
	       }

	   }
           else if (bquery)
	   {
	       szMessage += " and also the query set";
           }
 
           String szlogword ="";
	   if (theDataSet.btakelog)
	   {
	      szlogword ="log base two ";
	   }
           szMessage +=  ".\n\n The first four columns of the table are: \n"
                     + "*  Selected - contains a 'Yes' if the gene is part the GO Category or Query "
	             +"Set that the ";

	   if (theDataSet.bkmeans)
	   {
               szMessage += "K-means clusters ";
	   }
	   else
	   {
               szMessage += "profiles or cluster of profiles ";
	   }

              szMessage += 
		  "were ordered by, otherwise is blank.\n";
	      if (theDataSet.bkmeans)
	      {
                 szMessage +=  "*  Weight - Weight of the gene assignment, which is 1/(number of K-means clusters "+
                               "to which the gene was assigned)\n";
	      }
	      else
	      {
                 szMessage +=  "*  Weight - Weight of the gene assignment, which is 1/(number of profiles to which "+
			      "the gene was assigned)\n";
	      }
                  
              szMessage+="*  "+theDataSet.szGeneHeader+" - The name of the gene.\n"
                          +"*  "+theDataSet.szProbeHeader+" - The spot ID(s) associated with the gene.\n"
                          +"The remaining columns contain the "+szlogword+
                          "expression change relative to the first time point.\n\n"+
                          "Note:\n"
	                  +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
                          +"+Using the 'Copy Table' the entire table can be copied to the clipboard, or "+
                          "with the 'Copy Gene Names' button just the gene names.\n"
		          +"+Using the 'Save Table' button the entire table can be saved, or using the "+
			  "'Save Gene Names' button just the gene names.\n"+
                          "+'Chromosome View' button plots the genes in the table on the Chromosome Viewer and makes "+
                          "the viewer visible if not already so. If the button is inactive then no chromosome locations "+
		      "were specified.";

             Util.renderDialog(theFrame,szMessage);
       }
    }
}

