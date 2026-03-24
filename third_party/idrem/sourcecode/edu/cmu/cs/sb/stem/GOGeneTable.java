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
 * Class responsible for displaying the genes indicated to be assigned to 
 * a GO category in a gene table
 */
public class GOGeneTable extends JPanel implements ActionListener 
{
    private boolean DEBUG = false;

    ChromFrame cf;
    STEM_DataSet theDataSet;
    int nprofilecluster; 
    ArrayList clusterprofiles;
    String szHeaderInfo;
    String szGoID;
    String szGoCombined;
    boolean bqueryset;
    Vector columnNames;
    Vector tabledata;
    JButton saveButton;
    JButton savenamesButton;
    JButton copyButton;
    JButton copynamesButton;
    JButton chromButton;
    TableSorter sorter; 
    HashSet htinames;
    String szTitle;
    final static Color bg = Color.white;
    final static Color fg = Color.black;
    JFrame theFrame;
    boolean bjustgo;

    /**
     * Class constructor renders the table
     */
    public GOGeneTable(JFrame theFrame, STEM_DataSet theDataSet, String szGoID, String szGoCombined,
                       int nprofilecluster, String szHeaderInfo, ArrayList clusterprofiles,
                       boolean bqueryset, HashSet htinames,
                       String szTitle,boolean bjustgo, ChromFrame cf) 
    {
        this.theFrame = theFrame;
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
        setBackground(bg);
        setForeground(fg);
        this.bqueryset = bqueryset;
	this.bjustgo = bjustgo;
        this.theDataSet = theDataSet;
        this.nprofilecluster = nprofilecluster;
        this.szHeaderInfo = szHeaderInfo;  
        this.clusterprofiles = clusterprofiles;
        this.htinames = htinames;
        this.szGoID = szGoID;
        this.szGoCombined = szGoCombined;
        this.szTitle  = szTitle;
	this.cf = cf;
        int numcols; 

        columnNames = new Vector();

	if (bjustgo)
	{
            numcols = 2;
	}
	else
	{
           numcols = 4;
           columnNames.add("Selected");
           columnNames.add("Weight");
        }
        columnNames.add(theDataSet.szGeneHeader);
        columnNames.add(theDataSet.szProbeHeader);

        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
	{
            columnNames.add(""+theDataSet.dsamplemins[ncolindex]);
	}

        tabledata = new Vector();

        int ngeneindex = 0;

        if (clusterprofiles == null)
        {
           loadTable(theDataSet.profilesAssigned[nprofilecluster]);
        }
        else
        {
           for (int nindex = 0; nindex < clusterprofiles.size(); nindex++)
           {
              STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) clusterprofiles.get(nindex);
              loadTable(theDataSet.profilesAssigned[pr.nprofileindex]);     
           }
        }

        sorter = new TableSorter(new TableModelST(tabledata, columnNames));
        final JTable table = new JTable(sorter);
        sorter.setTableHeader(table.getTableHeader());
        table.setPreferredScrollableViewportSize(new Dimension(800, 
				     Math.min((table.getRowHeight()+table.getRowMargin())*
					      table.getRowCount(),300)));

        TableColumn column;
	if (bjustgo)
	{
           column = table.getColumnModel().getColumn(0);
           column.setPreferredWidth(200);
           column = table.getColumnModel().getColumn(1);
           column.setPreferredWidth(100);
	}
	else
	{
           column = table.getColumnModel().getColumn(0);
           column.setPreferredWidth(40);
           column = table.getColumnModel().getColumn(1);
           column.setPreferredWidth(40);
           column = table.getColumnModel().getColumn(2);
           column.setPreferredWidth(200);
           column = table.getColumnModel().getColumn(3);
           column.setPreferredWidth(150);
	}

        for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++)
	{
            column = table.getColumnModel().getColumn(ncolindex+numcols);
            column.setPreferredWidth(45);
	}

        //Create the scroll pane and add the table to it.
        JScrollPane scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);


        copyButton = new JButton("Copy Table",
                                Util.createImageIcon("Copy16.gif"));
        copyButton.setActionCommand("copy");
        copyButton.setMinimumSize(new Dimension(800,20));
        copyButton.addActionListener(this);

        saveButton = new JButton("Save Table", Util.createImageIcon("Save16.gif"));
        saveButton.setActionCommand("save");
        saveButton.setMinimumSize(new Dimension(800,20));
        saveButton.addActionListener(this);

        savenamesButton = new JButton("Save Gene Names", Util.createImageIcon("Save16.gif"));
        savenamesButton.setActionCommand("savenames");
        savenamesButton.setMinimumSize(new Dimension(800,20));
        savenamesButton.addActionListener(this);

        copynamesButton = new JButton("Copy Gene Names",
                                Util.createImageIcon("Copy16.gif"));
        copynamesButton.setActionCommand("copynames");
        copynamesButton.setMinimumSize(new Dimension(800,20));
        copynamesButton.addActionListener(this);

        chromButton = new JButton("Chromosome View");

	if ((cf==null)||(cf.genomeParser==null)||
	    (cf.genomeParser.szchromval.equals("")))
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

	buttonPanel.add(copynamesButton);
        buttonPanel.add(savenamesButton);
	buttonPanel.add(chromButton);

        JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpButton.addActionListener(this);
        helpButton.setActionCommand("help");
        buttonPanel.add(helpButton);
	buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel);
    }

    /**
     * Responds to interface actions on the table
     */
    public void actionPerformed(ActionEvent e)
    {
      String szCommand = e.getActionCommand();

      if (szCommand.equals("chromview"))
      {
         chromView();
      }
      else if (szCommand.equals("copynames"))
      {
	 writenamesToClipboard();
      }
      else if (szCommand.equals("copy"))
      {
         writeToClipboard();
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
		    printGeneNames(pw);
		}
                pw.close();
	     }
           }
           catch (FileNotFoundException ex)
           {
              ex.printStackTrace(System.out);
          }
       }
       else if (szCommand.equals("help"))
       {

          String szMessage = "The table contains genes assigned to the selected GO Category and ";

           if (clusterprofiles == null)
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
           else if (bqueryset)
	   {
	       szMessage += " and also the query set";
           }
 
           String szlogword ="";
	   if (theDataSet.btakelog)
	   {
	      szlogword ="log base two ";
	   }
          
	       if (bjustgo)
	       {
		   szMessage += ".\n\n The first two columns of the table are: \n";
	       }
	       else
	       {
		   if (theDataSet.bkmeans)
		   {
                     szMessage += ".\n\n The first four columns of the table are: \n"
                     + "*  Selected - contains a 'Yes' if the gene is part the GO Category or Query "
                     +"Set that the K-means clusters "+
		      "were ordered by, otherwise is blank.\n"
                          +"*  Weight - Weight of the gene assignment, which is 1/(number of K-means clusters to which "+
                          "the gene was assigned)\n";
		   }
		   else
		   {
                     szMessage += ".\n\n The first four columns of the table are: \n"
                     + "*  Selected - contains a 'Yes' if the gene is part the GO Category or Query "
                     +"Set that the profiles or cluster of profiles "+
		      "were ordered by, otherwise is blank.\n"
                          +"*  Weight - Weight of the gene assignment, which is 1/(number of profiles to which "+
                          "the gene was assigned)\n";
		   }

	       }
                    szMessage += "*  "+theDataSet.szGeneHeader+" - The name of the gene.\n"
			+"*  "+theDataSet.szProbeHeader+" - The spot ID(s) associated with the gene.\n\n";
		    if (bjustgo)
		     {
                         if(columnNames.size()==3)
		         {
                            szMessage += "The remaining column contains the "+szlogword+
			      "expression change relative to the first time point.\n\n";
		         }
                     }
                     else
		     {
                          szMessage += "The remaining columns contain the "+szlogword+
			      "expression change relative to the first time point.\n\n";
		      }

                      szMessage +=
                          "Note:\n"
	                   +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
                           +"Using the 'Copy Table' the entire table can be copied to the clipboard, or "+
                           "with the 'Copy Gene Names' button just the gene names.\n"
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


    /**
     * Writes the contents of the table to a file
     */
    public void printFile(PrintWriter pw)
    {
       if (!bjustgo)
       {
          if (clusterprofiles == null)
          {
	     pw.print(szTitle+" (");
             for (int nindex = 0; nindex < theDataSet.numcols-1; nindex++)
             {
                pw.print(theDataSet.modelprofiles[nprofilecluster][nindex]+",");
             }
             pw.println(theDataSet.modelprofiles[nprofilecluster][theDataSet.numcols-1]+")");
          }
          else
          {
             pw.println(szTitle+":");
             for (int nprofileindex = 0; nprofileindex < clusterprofiles.size(); nprofileindex++)
             {
                STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) clusterprofiles.get(nprofileindex);
                pw.print("\t Profile "+pr.nprofileindex+" (");
                for (int nindex = 0; nindex < theDataSet.numcols-1; nindex++)
                {
                   pw.print(theDataSet.modelprofiles[pr.nprofileindex][nindex]+",");
                }
                pw.println(theDataSet.modelprofiles[pr.nprofileindex][theDataSet.numcols-1]+")");
             }
	  }
       }
      
       pw.println("#genes category\t#genes assigned\t#genes expected\t"+
                   "#genes enriched\tp-value\tcorrected p-value");
       pw.println(szHeaderInfo);

       pw.println("-------");

       int ncnsminus1 = columnNames.size()-1;
       for (int ncol = 0; ncol < ncnsminus1; ncol++)
       {
          pw.print(columnNames.get(ncol) +"\t");
       }
       pw.println(columnNames.get(ncnsminus1));

       for (int nrow = 0; nrow <tabledata.size(); nrow++)
       {
          for (int ncol = 0; ncol < ncnsminus1; ncol++)
          {
              pw.print(sorter.getValueAt(nrow, ncol)+"\t");
          }
          pw.println(sorter.getValueAt(nrow,ncnsminus1));
       }
    }

    /**
     * Writes the contents of the table to a clipboard
     */
    public void writeToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();

       int ncnsminus1 = columnNames.size()-1;
       for (int ncol = 0; ncol < ncnsminus1; ncol++)
       {
          sbuf.append(columnNames.get(ncol) +"\t");
       }
       sbuf.append(columnNames.get(ncnsminus1)+"\n");

       for (int nrow = 0; nrow <tabledata.size(); nrow++)
       {
          for (int ncol = 0; ncol < ncnsminus1; ncol++)
          {
              sbuf.append(sorter.getValueAt(nrow, ncol)+"\t");
          }
          sbuf.append(sorter.getValueAt(nrow,ncnsminus1)+"\n");
       }

       	// get the system clipboard
       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }

    /**
     * Writes just the gene names to a file
     */
    public void printGeneNames(PrintWriter pw)
    {
       int nsize = tabledata.size();
       int ncol;

       if (bjustgo)
       {
	   ncol = 0;
       }
       else
       {
           ncol = 2;
       }

       for (int nrow = 0; nrow < nsize; nrow++)
       {
          pw.println(sorter.getValueAt(nrow, ncol));
       }
    }

    /**
     * Copies just the gene names to the clipboard
     */ 
    public void writenamesToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();

       int nsize = tabledata.size();
       int ncol;

       if (bjustgo)
       {
	   ncol = 0;
       }
       else
       {
           ncol = 2;
       }

       for (int nrow = 0; nrow < nsize; nrow++)
       {
          sbuf.append(sorter.getValueAt(nrow, ncol)+"\n");
       }

       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }

    /**
     * Calls the chromosome viewer to draw and display gene on this table
     */
    public void chromView()
    {	
       int nsize = tabledata.size();
       int ncol;

       if (bjustgo)
       {
	   ncol = 0;
       }
       else
       {
           ncol = 2;
       }

       String[] chromgeneNames = new String[nsize];
       for(int nrow = 0; nrow < chromgeneNames.length; nrow++)
       {
	  chromgeneNames[nrow] = (String) sorter.getValueAt(nrow, ncol);
       }

       cf.drawGenes(chromgeneNames);
       cf.setVisible(true);    
    }

    /**
     *  Adds to the table the genes that should be displayed on the table
     */
    public void loadTable(ArrayList profileBestIndex)
    {
        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);

       int nbestsize = profileBestIndex.size();
       for (int nindex = 0; nindex < nbestsize; nindex++)
       {
          Vector rec = new Vector(3); 
          int ngeneindex = ((Integer) profileBestIndex.get(nindex)).intValue();
 
          HashSet goList;
          String szid;

	  szid = theDataSet.genenames[ngeneindex]; 
          goList = theDataSet.tga.labelsForID(szid);


          if ((goList.size()>=1)&&(goList.contains(szGoID)))
          {              
	      if (((!bqueryset)&& (htinames == null))|| 
                  ((htinames != null) && (htinames.contains(theDataSet.genenames[ngeneindex]))) ||
                  ((htinames == null) &&
		  (theDataSet.tga.isOrder(theDataSet.genenames[ngeneindex]))))
	     {
		 if (!bjustgo)
		 {
                    if ((bqueryset)|| 
                        (htinames != null) ||
                        (theDataSet.tga.isOrder(theDataSet.genenames[ngeneindex])))
		    {
		       rec.add("Yes");
                    }
                    else
	            {
		        rec.add("");
	            }
                    rec.add(nf2.format(1.0/(double)theDataSet.bestassignments[ngeneindex].size()));
		 }
                 rec.add(theDataSet.genenames[ngeneindex]);
                 rec.add(theDataSet.probenames[ngeneindex]);
                 for (int ncol = 0; ncol < theDataSet.numcols; ncol++)
	         {
                    if (theDataSet.pmavalues[ngeneindex][ncol]==0)
	            {
	               rec.add("");
                    }
                    else
	            {
                       rec.add(nf2.format(theDataSet.data[ngeneindex][ncol]));
	            }
	         }
                 tabledata.add(rec);
	     }
          }
       }
    }
}

