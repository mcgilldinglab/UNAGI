package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.chromviewer.*;
import edu.cmu.cs.sb.core.*;
import javax.swing.*;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import javax.swing.table.*;
import javax.swing.event.*;
import java.awt.Dimension;
import java.awt.GridLayout;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.*;
import java.text.NumberFormat;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import java.awt.datatransfer.*;

/**
 * Encapsulates a Gene Ontology enrichment analysis
 * where the enrichment calculation is based on the expected number of genes
 * to the profile
 */
public class GOTableExpected extends JPanel implements ActionListener
{
    private boolean DEBUG = false;

    final String szqueryset;
    JFrame theFrame;
    STEM_DataSet theDataSet;
    int nprofile; 

    GoAnnotations.GoResults tgr;
    final static Color bg = Color.white;
    final static Color fg = Color.black;
    JButton saveButton,copyButton;
    ChromFrame cf;

    String[][] tabledata= null;
    String[] columnNames = null;
    ArrayList clusterArray =null;
    boolean bqueryset;    
    TableSorter sorter;
    HashSet htinames;
    String szTitle;


    /**
     * Constructor for the GO table, renders the table
     */
   public GOTableExpected(JFrame theFrame,STEM_DataSet theDataSet, int nprofile,ArrayList clusterArray, 
		      boolean bqueryset,HashSet htinames,String szTitle,final ChromFrame cf) 
    {
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
        setBackground(bg);
        setForeground(fg);
	this.bqueryset =bqueryset;
        this.theFrame = theFrame;
        this.theDataSet = theDataSet;
        this.nprofile = nprofile;
        this.clusterArray = clusterArray;
        this.htinames = htinames;
        this.szTitle = szTitle;
	this.cf = cf;

        if (htinames != null)
        {
            szqueryset = szTitle.substring(15);
	}
	else if (bqueryset)
	{
	    szqueryset = "Query Set and ";
	}
        else
	{
            szqueryset = "";
	}
        int numcols = 9;
        columnNames = new String[numcols];
	
        columnNames[0] = "Category ID";
        columnNames[1] = "Category Name";
        columnNames[6] = "p-value";
        columnNames[7] = "Corrected\np-value";
        columnNames[2] = "#Genes\nCategory";
        columnNames[3] = "#Genes\nAssigned";
        columnNames[4] = "#Genes\nExpected\n";
        columnNames[5] = "#Genes\nEnriched";
        columnNames[8] = "Fold";

        if (clusterArray == null)
        {
           tgr = theDataSet.tga.getCategory(theDataSet.profilesAssigned[nprofile],
		     theDataSet.genenames, theDataSet.bestassignments,bqueryset,htinames,
                     theDataSet.expectedassignments[nprofile]);
        }
        else
        {
           tgr = theDataSet.tga.getCategory(clusterArray,  
                        theDataSet.profilesAssigned,theDataSet.genenames, 
			    theDataSet.bestassignments,bqueryset,htinames);
         }
        
         tabledata = new String[tgr.goRecArray.length][columnNames.length];

         NumberFormat nf3 = NumberFormat.getInstance(Locale.ENGLISH);
         nf3.setMinimumFractionDigits(3);
         nf3.setMaximumFractionDigits(3);
        
         NumberFormat nf1 =  NumberFormat.getInstance(Locale.ENGLISH);
         nf1.setMinimumFractionDigits(1);
         nf1.setMaximumFractionDigits(1);
         nf1.setGroupingUsed(false);

         for (int nindex = 0; nindex < tgr.goRecArray.length; nindex++)
	 {
            tabledata[nindex][0] = tgr.goRecArray[nindex].szgoid;
            tabledata[nindex][1] = tgr.goRecArray[nindex].szgocategory;
            tabledata[nindex][6] = Util.doubleToSz(tgr.goRecArray[nindex].dpvaluebinom);
            if (tgr.goRecArray[nindex].dcorrectedpvaluebinom < 0.001)
	    {
	       tabledata[nindex][7] = Util.doubleToSz(tgr.goRecArray[nindex].dcorrectedpvaluebinom);
	    }
            else 
	    {
               tabledata[nindex][7] = nf3.format(tgr.goRecArray[nindex].dcorrectedpvaluebinom);
	    }

            tabledata[nindex][3]=nf1.format(tgr.goRecArray[nindex].dcategoryselect); 
            tabledata[nindex][2] = "" + (int) tgr.goRecArray[nindex].dcategoryall;
            double dexpected = tgr.goRecArray[nindex].dcategoryall
		  * theDataSet.expectedassignments[nprofile]/tgr.dall;
	    tabledata[nindex][4] = nf1.format(dexpected);
                
	    double denriched = tgr.goRecArray[nindex].dcategoryselect - dexpected;
            tabledata[nindex][5] = nf1.format(denriched);
            double dval = Double.parseDouble(tabledata[nindex][5]);
            if (dval > 0)
            {
               tabledata[nindex][5] = "+" + tabledata[nindex][5];
            }
            else if (dval == 0)
	    {
	       tabledata[nindex][5] = "0.0";
	    }
	    tabledata[nindex][8] = nf1.format(tgr.goRecArray[nindex].dcategoryselect/dexpected);
	 }

	 sorter = new TableSorter(new TableModelST(tabledata, columnNames));
	 final JTable table = new JTable(sorter);

	 MultiLineHeaderRenderer renderer = new MultiLineHeaderRenderer(sorter);
	 Enumeration enumv = table.getColumnModel().getColumns();

	 while (enumv.hasMoreElements()) 
         {
	     ((TableColumn)enumv.nextElement()).setHeaderRenderer(renderer);
	 }  
         sorter.setTableHeader(table.getTableHeader());
           
         table.setPreferredScrollableViewportSize(new Dimension(800, 
                               Math.min((table.getRowHeight()+table.getRowMargin())*
					table.getRowCount(),400)));
          
         TableColumn column;
         column = table.getColumnModel().getColumn(0);
         column.setPreferredWidth(30);
         column = table.getColumnModel().getColumn(1);
         column.setPreferredWidth(200);
         column = table.getColumnModel().getColumn(2);
         column.setPreferredWidth(20);
         column = table.getColumnModel().getColumn(3);
         column.setPreferredWidth(20);
         column = table.getColumnModel().getColumn(4);
         column.setPreferredWidth(20);
         column = table.getColumnModel().getColumn(5);
         column.setPreferredWidth(20);
         column = table.getColumnModel().getColumn(6);
         column.setPreferredWidth(20);
         column = table.getColumnModel().getColumn(7);
         column.setPreferredWidth(20);      
         column = table.getColumnModel().getColumn(8);
         column.setPreferredWidth(20);    
 
         //Create the scroll pane and add the table to it.
         JScrollPane scrollPane = new JScrollPane(table);

         //Add the scroll pane to this panel.
         add(scrollPane);

         copyButton = new JButton("Copy Table",
                                Util.createImageIcon("Copy16.gif"));
         copyButton.setActionCommand("copy");
         copyButton.setMinimumSize(new Dimension(800,20));
         copyButton.addActionListener(this);

         saveButton = new JButton("Save Table", 
                                Util.createImageIcon("Save16.gif"));
         saveButton.setActionCommand("save");
         saveButton.setMinimumSize(new Dimension(800,20));
         saveButton.addActionListener(this);

         table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
         ListSelectionModel rowSM = table.getSelectionModel();
         final STEM_DataSet finaltheDataSet = theDataSet;
         final int finalnprofile = nprofile;
         final ArrayList finalclusterArray = clusterArray;

         final TableSorter finalsorter = sorter;
         final boolean bfinalqueryset = bqueryset;
         final HashSet fhtinames = htinames;
           
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
                         int selectedRow = lsm.getMinSelectionIndex();
                        
                         //id then title
                         String szGoCombined = finalsorter.getValueAt(selectedRow,0) +
			           " ("+finalsorter.getValueAt(selectedRow,1)+")";
                         String sznewTitle = "Gene List for "+szGoCombined +
                                          " genes in "+szqueryset;
	 
                         if (fhtinames == null)
			 {
                            if (finalclusterArray == null)
                            {
				sznewTitle += "Profile " + finalnprofile;
			    }
                            else
                            {
				sznewTitle += "Cluster " + finalnprofile;
                            }
			 }

                         JFrame frame = new JFrame(sznewTitle);
                         frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                        
                         String szHeaderInfo = finalsorter.getValueAt(selectedRow,2)+"\t"+
		    	 finalsorter.getValueAt(selectedRow,3)+"\t"+
			 finalsorter.getValueAt(selectedRow,4)+"\t"+
			 finalsorter.getValueAt(selectedRow,5)+"\t"+
                         finalsorter.getValueAt(selectedRow,6)+"\t"+
			 finalsorter.getValueAt(selectedRow,7)+"\t"+
			 finalsorter.getValueAt(selectedRow,8);
                  
                         GOGeneTable newContentPane = new GOGeneTable(frame,finaltheDataSet, 
							(String) finalsorter.getValueAt(selectedRow,0),
                                                                     szGoCombined,
                                                                     finalnprofile, 
                                                                     szHeaderInfo,
								     finalclusterArray,
									 bfinalqueryset,fhtinames,sznewTitle,false,cf);
                     
                         newContentPane.setOpaque(true); //content panes must be opaque
                         frame.setContentPane(newContentPane);
                         frame.setLocation(40,150);
                         //Display the window.
                         frame.pack();
                         frame.setVisible(true);
		       }
		    });
		 }
	     }
	 });

         JPanel buttonPanel = new JPanel();
         buttonPanel.setBackground(Color.white);
         buttonPanel.add(copyButton);
         buttonPanel.add(saveButton);

         JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
         helpButton.addActionListener(this);
         helpButton.setActionCommand("help");
         buttonPanel.add(helpButton);
	 buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
         add(buttonPanel);
    }
    

    /**
     * Writes the contents of the table to a file
     */
    public void printFile(PrintWriter pw)
    {      
       if (clusterArray == null)
       {
           pw.print(szTitle+" (");

           for (int nindex = 0; nindex < theDataSet.numcols-1; nindex++)
           {
              pw.print(theDataSet.modelprofiles[nprofile][nindex]+",");
           } 
           pw.println(theDataSet.modelprofiles[nprofile][theDataSet.numcols-1]+")");
       }
       else
       {
           pw.println(szTitle+":");

           for (int nprofileindex = 0; nprofileindex < clusterArray.size(); nprofileindex++)
           {
              STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) clusterArray.get(nprofileindex);
              pw.print("\t Profile "+pr.nprofileindex+" (");
              for (int nindex = 0; nindex < theDataSet.numcols-1; nindex++)
              {
                 pw.print(theDataSet.modelprofiles[pr.nprofileindex][nindex]+",");
              } 
              pw.println(theDataSet.modelprofiles[pr.nprofileindex][theDataSet.numcols-1]+")");
           } 
       }
       
       pw.println("-------");

       for (int ncol = 0; ncol < columnNames.length; ncol++)
       {
	   String szcol = columnNames[ncol];
           for (int nch = 0; nch < szcol.length(); nch++)
	   {
	       char ch = szcol.charAt(nch);
	       if (ch == '\n')
	       {
                   pw.print(" ");
	       }
               else
	       {
                   pw.print(ch);
	       }
	   }
           if (ncol < columnNames.length-1)
	   {
              pw.print("\t");
	   }
       }
       pw.println();

       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
          for (int ncol = 0; ncol < tabledata[nrow].length-1; ncol++)
          {
             pw.print(tabledata[nrow][ncol]+"\t");
          }
          pw.println(tabledata[nrow][tabledata[nrow].length-1]);
       } 
    }

    /**
     * Writes the contents of the table to the clipboard
     */
    public void writeToClipboard() 
    {
       StringBuffer sbuf =new StringBuffer();

       for (int ncol = 0; ncol < columnNames.length; ncol++)
       {
	   String szcol = columnNames[ncol];
           for (int nch = 0; nch < szcol.length(); nch++)
	   {
	       char ch = szcol.charAt(nch);
	       if (ch == '\n')
	       {
                   sbuf.append(" ");
	       }
               else
	       {
                   sbuf.append(ch);
	       }
	   }
           if (ncol < columnNames.length-1)
	   {
              sbuf.append("\t");
	   }
       }
       sbuf.append("\n");

       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
          for (int ncol = 0; ncol < tabledata[nrow].length-1; ncol++)
          {
             sbuf.append(tabledata[nrow][ncol]+"\t");
          }
          sbuf.append(tabledata[nrow][tabledata[nrow].length-1]+"\n");
       } 

       	// get the system clipboard
       Clipboard systemClipboard =
		       Toolkit.getDefaultToolkit().getSystemClipboard();
       Transferable transferableText = new StringSelection(sbuf.toString());
       systemClipboard.setContents(transferableText, null);
    }


    /**
     * Responds to interface actions on the interface
     */
    public void actionPerformed(ActionEvent e) 
    {
       String szCommand = e.getActionCommand();
       if (szCommand.equals("copy"))
       {
	   writeToClipboard();
       }
       else if (szCommand.equals("save"))
       {
          try
          {
	     int nreturnVal = Util.theChooser.showSaveDialog(this);
             if (nreturnVal == JFileChooser.APPROVE_OPTION) 
             {
                File f = Util.theChooser.getSelectedFile();
                PrintWriter pw = new PrintWriter(new FileOutputStream(f)); 
                printFile(pw);
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
           NumberFormat nf2 =  NumberFormat.getInstance(Locale.ENGLISH);
           nf2.setMinimumFractionDigits(2);
           nf2.setMaximumFractionDigits(2);
           String szMessage = "The table contains a Gene Ontology (GO) enrichment analysis "+
                              "for the set of genes in the selected profile"+
                              " as compared to what would be expected if the number of genes expected to be assigned to this "+
                              "set based on a permutation test were randomly selected "
	                      + "among the entire set of unique genes on the microarray.\n"; 

           szMessage += "\nThe table has nine columns:" +"\n"+
                               "1.  Category ID - the ID for the category\n"+
                              "2.  Category Name - the name for the category \n"+
                               "3.  #Genes Category - the number of genes on the array annotated as "
                               +"belonging to this GO category\n"+
                              "4.  #Genes Assigned - the number of genes of this GO category that were assigned "+
                               "to this profile or cluster\n"+
                              "5.  #Genes Expected - the number of genes of this GO category expected to be "+ 
                              "assigned to this set of genes.  This is computed as "+
                              "(#Expected_In_Profile)*(#Genes_Category)/(#Total_Unique_Genes_on_Array).\n"+
                               "For this set  #Expected_In_Profile = "+nf2.format(theDataSet.expectedassignments[nprofile])
                               +" and for the data  #Total_Unique_Genes_on_Array="+((int)tgr.dall)+".\n"+
                               "6.  #Genes Enriched - The difference between the #Genes Assigned and the "+
                                     "#Genes Expected\n"+
                              "7.  p-value - the uncorrected probability of seeing greater than "+
                              "#Genes_Assigned assigned to this set from this GO category than "+
                               "were observed by random chance.  "+
                              "This is computed based on the Binomial distribution with parameters: "+
                              "n=#Genes_Category, and p=((#Expected_In_Profile)/(#Total_Unique_Genes_on_Array))\n"+
                              "8.  Corrected p-value - the p-value corrected using a Bonferonni correction "+
                              "for multiple GO categories being "+
                              "tested simultaneously.  This corrected p-value is the original p-value"+
                              " divided by the number of GO Categories for which the number of genes on the "+
                              "array is above the GO Analysis minimum number of genes parameter.\n"+
                             "9. Fold - fold enrichment that is the number of genes assigned divided by expected\n\n"+
                              "Note:\n"+
                              "+Clicking on a row of the table displays the list of genes in "+
                              "the set being analyzed that are by of the GO category of the row clicked.\n"
                              +"+The table can be sorted by any of the columns by clicking on the desired "+
                              " column's header.\n"+
                              "+Using the 'Copy Table' the entire table can be copied to the clipboard.\n"
	                      +"+Using the 'Save Table' button the entire table can be saved.\n"; 
                              
	   Util.renderDialog(theFrame,szMessage);
       }
    }
}

