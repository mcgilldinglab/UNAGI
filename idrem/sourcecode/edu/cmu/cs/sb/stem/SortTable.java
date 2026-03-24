package edu.cmu.cs.sb.stem;

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
import edu.umd.cs.piccolox.PFrame;
import java.awt.datatransfer.*;

/**
 * Class for the interface to select how profiles or cluster of profiles should be sorted
 */
public class SortTable extends JPanel implements ActionListener
{
    private boolean DEBUG = false;

    STEM_DataSet theDataSet;
    int nprofile; 
    JTable table = null;
    JScrollPane scrollPane =null;
    final static Color bg = Color.white;
    final static Color fg = Color.black;
    String[][] tabledata   = null;
    String[] columnNames   = null;
    ArrayList clusterArray = null;
    MAINGUI2 maingui;
    GoAnnotations tga;
    TableSorter sorter;
    JFrame theFrame;
    JRadioButton actualButton, expectedButton;
    ListSelectionModel rowSM;
    JFrame defineframe;

    /**
     * Class constructor - renders the interface
     */
    public SortTable(JFrame theFrame,MAINGUI2 maingui, GoAnnotations tga,STEM_DataSet theDataSet) 
    {
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
        setBackground(bg);
        setForeground(fg);
        this.theFrame = theFrame;
        this.maingui = maingui;
        this.tga = tga;
        this.theDataSet = theDataSet;

        if (tga.bcluster)
	{
           GoAnnotations.RecIDpval[] theRecIDpval;
           columnNames = new String[3];
	
           columnNames[0] = "Category ID";
           columnNames[1] = "Category Name";
           columnNames[2] = "Min p-value";
	   theRecIDpval = tga.theClusterRecIDpval;
           tabledata = new String[theRecIDpval.length][columnNames.length];
       
           for (int ncategory = 0; ncategory < theRecIDpval.length; ncategory++)
	   {
	      tabledata[ncategory][0] = theRecIDpval[ncategory].szid;
              tabledata[ncategory][2] = 
                      Util.doubleToSz(theRecIDpval[ncategory].dpval);
              String szgolabel = "";
              String sztemp = ((GoAnnotations.Rec) 
			     tga.htGO.get(tabledata[ncategory][0])).sztermName;

              if (sztemp != null)
	      {
	         szgolabel = sztemp;
	      }
              tabledata[ncategory][1] = szgolabel;
	   }
	}
        else
	{
	   GoAnnotations.RecIDpval2[] theRecIDpval = tga.theRecIDpval;
	   if (theDataSet.bkmeans)
	   {
              columnNames = new String[3];
	   }
	   else
	   {
              columnNames = new String[4];
              columnNames[3] = "Min p-value\n(expected size)";
	   }

           columnNames[0] = "Category ID";
           columnNames[1] = "Category Name";
           columnNames[2] = "Min p-value\n(actual size)";
       
           tabledata = new String[theRecIDpval.length][columnNames.length];
       
           for (int ncategory = 0; ncategory < theRecIDpval.length; ncategory++)
	   {
	      tabledata[ncategory][0] = theRecIDpval[ncategory].szid;
              tabledata[ncategory][2] = 
                      Util.doubleToSz(theRecIDpval[ncategory].dpvalhyper);
	      if (!theDataSet.bkmeans)
	      {
                 tabledata[ncategory][3] = 
		     Util.doubleToSz(theRecIDpval[ncategory].dpvalbinom);
	      }
              String szgolabel = "";

              String sztemp = ((GoAnnotations.Rec) 
			     tga.htGO.get(tabledata[ncategory][0])).sztermName;

              if (sztemp != null)
	      {
	         szgolabel = sztemp;
	      }
              tabledata[ncategory][1] = szgolabel;
	   }
	}

        sorter = new TableSorter(new TableModelST(tabledata,columnNames));
        table = new JTable(sorter);
                     
        TableColumn column;
        column = table.getColumnModel().getColumn(0);
        column.setPreferredWidth(100);
        column = table.getColumnModel().getColumn(1);
        column.setPreferredWidth(250);
        column = table.getColumnModel().getColumn(2);
        column.setPreferredWidth(100);
        if ((!tga.bcluster)&&(!theDataSet.bkmeans))
        {
           column = table.getColumnModel().getColumn(3);
           column.setPreferredWidth(100);
        }
        MultiLineHeaderRenderer renderer = new MultiLineHeaderRenderer(sorter);
        Enumeration enumv = table.getColumnModel().getColumns();

        while (enumv.hasMoreElements()) 
        {
           ((TableColumn)enumv.nextElement()).setHeaderRenderer(renderer);
        }  
        sorter.setTableHeader(table.getTableHeader());

        //Create the scroll pane and add the table to it.
        scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);
 
        //table.setCellSelectionEnabled(true);
        table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
        rowSM = table.getSelectionModel();
        final STEM_DataSet finaltheDataSet = theDataSet;
        final boolean bfcluster = tga.bcluster;
        final MAINGUI2 finalmaingui = maingui;

        rowSM.addListSelectionListener(new ListSelectionListener() {
           public void valueChanged(ListSelectionEvent e) {
              //Ignore extra messages.
              if (e.getValueIsAdjusting()) return;

              ListSelectionModel lsm = (ListSelectionModel)e.getSource();
              if (lsm.isSelectionEmpty()) 
              {
                 return;
              } 
              else 
              {
                 int selectedRow = lsm.getMinSelectionIndex();
                        
                 String szSelectedGO = (String) 
                                      sorter.getValueAt(selectedRow,0);

                 finaltheDataSet.tga.szSelectedGO = szSelectedGO;
                 finaltheDataSet.tga.bcluster = bfcluster;
                        
                 if ((finaltheDataSet.bkmeans)||(bfcluster)||(actualButton.isSelected()))
		 {
                    finaltheDataSet.tga.szsortcommand = "go";
		 }
		 else
	       	 {
		    finaltheDataSet.tga.szsortcommand = "expgo";
		 }

		 finalmaingui.drawmain();
              }
           }
        });
	
        JPanel buttonPanel = new JPanel();
        buttonPanel.setBackground(Color.white);

	if (!theDataSet.bkmeans)
	{
           JButton defaultButton = new JButton("Default Order");
           defaultButton.setActionCommand("default");
           defaultButton.addActionListener(this);
           buttonPanel.add(defaultButton);
	}
        JButton defineButton = new JButton("Define Gene Set...");
        defineButton.setActionCommand("define");
        defineButton.addActionListener(this);


        JButton copyButton = new JButton("Copy Table",
                                Util.createImageIcon("Copy16.gif"));
        copyButton.setActionCommand("copy");
        copyButton.setMinimumSize(new Dimension(800,20));
        copyButton.addActionListener(this);

        JButton saveButton = new JButton("Save Table",Util.createImageIcon("Save16.gif"));
        saveButton.setActionCommand("save");
        saveButton.addActionListener(this);
  
        if (!theDataSet.tga.bcluster)
	{
	   JButton idButton,condButton;
           JButton sigButton, numButton,expButton;
	   JPanel buttonPanel3 = new JPanel();

	   if (!theDataSet.bkmeans)
	   {
              buttonPanel3.setBackground(Color.white);
              actualButton = new JRadioButton("actual size");
              actualButton.setBackground(Color.white);
              actualButton.setSelected(true);
              actualButton.setActionCommand("actualgo");
	      actualButton.addActionListener(this);
              expectedButton = new JRadioButton("expected size");
              expectedButton.setBackground(Color.white);
              expectedButton.setActionCommand("expectedgo");
	      expectedButton.addActionListener(this);
              ButtonGroup group = new ButtonGroup();
              group.add(actualButton);
              group.add(expectedButton);
              buttonPanel3.add(new JLabel("Order using enrichment p-values based on a profile's"));
              buttonPanel3.add(actualButton);
              buttonPanel3.add(expectedButton);
	      buttonPanel3.setMaximumSize(new Dimension(Integer.MAX_VALUE,200));
	      add(buttonPanel3);
	   }

           JPanel buttonPanel2 = new JPanel();
           buttonPanel2.setBackground(Color.white);
	   if (theDataSet.bkmeans)
	   {
              idButton = new JButton("Cluster ID");
	   }
	   else
	   {
              idButton = new JButton("Profile ID");
	   }

           idButton.setActionCommand("id");
           idButton.addActionListener(this);
           buttonPanel2.add(idButton);

	   if (!theDataSet.bkmeans)
	   {
              sigButton = new JButton("Significance");
              sigButton.setActionCommand("sig");
              sigButton.addActionListener(this);
              buttonPanel2.add(sigButton);
	   }

           numButton = new JButton("Number of Genes");
           numButton.setActionCommand("num");
           numButton.addActionListener(this);
           buttonPanel2.add(numButton);

	   if (!theDataSet.bkmeans)
	   {
              expButton = new JButton("Expected Number");
              expButton.setActionCommand("exp");
              expButton.addActionListener(this);
              buttonPanel2.add(expButton);
	   }
	   buttonPanel2.setMaximumSize(new Dimension(Integer.MAX_VALUE,200));
           add(buttonPanel2);
           table.setPreferredScrollableViewportSize(new Dimension(450,  
                          Math.min((table.getRowHeight()+table.getRowMargin())*
				   table.getRowCount(),250)));
	}
        else
	{
          table.setPreferredScrollableViewportSize(new Dimension(450, 300));
	}
           
        buttonPanel.add(defineButton);
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
     * Cloes the Define Gene Set window
     */
    public void closeDefineWindows()
    {
       if (defineframe != null)
       {
	  defineframe.setVisible(false);
	  defineframe.dispose();
          defineframe = null;
       } 
    }

    /**
     * Writes the contents of this table to a file
     */
    public void printFile(PrintWriter pw)
    {      

       if (tga.bcluster)
       {
          pw.println("Minimum p-value of GO enrichment in any cluster");       
       }
       else
       {
          pw.println("Minimum p-value of GO enrichment in any profile"); 
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
     * Copies the content of this table to the clipboard
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

    //////////////////////////////////////////////////////////

    /**
     * Responds to interace actions on this interface window
     */
    public void actionPerformed(ActionEvent e)
    {
	String szcommand = e.getActionCommand();

	if ((szcommand.equals("actualgo"))||(szcommand.equals("expectedgo")))
	{
           theDataSet.tga.bcluster = false;  //only on profile window
           theDataSet.tga.bactual = actualButton.isSelected();
           if ((theDataSet.tga.szsortcommand.equals("define"))&&(szcommand.equals("expectedgo")))
	   { 
	      //already sorting based on a query set based on actual enrichment, want to use expected now 
              theDataSet.tga.szsortcommand = "expdefine";
	      maingui.drawmain();
	   }
           else if ((theDataSet.tga.szsortcommand.equals("expdefine"))&&(szcommand.equals("actualgo")))
	   {
               //already sorting based on a query set based on expected enrichment, want to use actual now
               theDataSet.tga.szsortcommand = "define";
	       maingui.drawmain();
	   }
	   else if 
               (((theDataSet.tga.szsortcommand.equals("expgo"))&&(szcommand.equals("actualgo")))||
		((theDataSet.tga.szsortcommand.equals("go"))&&(szcommand.equals("expectedgo"))))
	   {
	      int selectedRow = rowSM.getMinSelectionIndex();
	      if (selectedRow >= 0)
	      {                 
                 String szSelectedGO = (String)  sorter.getValueAt(selectedRow,0);

                 theDataSet.tga.szSelectedGO = szSelectedGO;
                 theDataSet.tga.bcluster = false;  //only on profile window
              
                 if (szcommand.equals("actualgo"))
	         {
                    theDataSet.tga.szsortcommand = "go";
	         }
   	         else
	         {
  	            theDataSet.tga.szsortcommand = "expgo";
	         }
	         maingui.drawmain();
	      }
         }
      }
      if (szcommand.equals("default"))
      {
	   rowSM.clearSelection();
	   theDataSet.tga.szSelectedGO = null;
	   theDataSet.tga.szsortcommand = "default";
	   maingui.drawmain();
      }
      else if (szcommand.equals("sig"))
      {
	  rowSM.clearSelection();
	  theDataSet.tga.szSelectedGO = null;
	  theDataSet.tga.szsortcommand = "sig";
	  maingui.drawmain();
      }
      else if (szcommand.equals("num"))
      {
	  rowSM.clearSelection();
	  theDataSet.tga.szSelectedGO = null;
	  theDataSet.tga.szsortcommand = "num";
	  maingui.drawmain();
      }
      else if (szcommand.equals("exp"))
      {
         rowSM.clearSelection();
	 theDataSet.tga.szSelectedGO = null;
         theDataSet.tga.szsortcommand = "exp";
         maingui.drawmain();
      }
      else if (szcommand.equals("id"))
      {
	  rowSM.clearSelection();
	  theDataSet.tga.szSelectedGO = null;
	  theDataSet.tga.szsortcommand = "id";
	  maingui.drawmain();
       }
       else if (szcommand.equals("define"))
       {
          javax.swing.SwingUtilities.invokeLater(new Runnable() 
          {
             public void run() 
             {
	        if (defineframe == null)
		{
                   defineframe = new JFrame("Define Gene Set");
	           defineframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                   defineframe.setLocation(400,200);
                   DefineGeneSet newContentPane = new DefineGeneSet(defineframe,maingui,tga,
                                                                    rowSM,theDataSet.bkmeans);
                   newContentPane.setOpaque(true); 
                   //content panes must be opaque
                   defineframe.setContentPane(newContentPane);

                   //Display the window.
                   defineframe.pack();
		}
		else
		{
		   defineframe.setExtendedState(Frame.NORMAL);
		}
                defineframe.setVisible(true);
	     }
	  });
       }
       else if (szcommand.equals("copy"))
       {
          writeToClipboard();
       }
       else if (szcommand.equals("save"))
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
       else if (szcommand.equals("help"))
       {
           String szMessage;
	   if (theDataSet.bkmeans)
	   {
              szMessage =
                "K-means clusters boxes are ordered on the screen from top to bottom and left to right.  " +
		  "By default profiles clusters are ordered by ID.\n\n"+
               "Cluster boxes can be reordered from top to bottom and "+
	       "left to right by one of several criteria: \n"+
               "*  GO Category Gene Enrichment -- the K-means clusters can be reordered by "+
               "gene enrichment of genes annotated as belonging to any GO Category.  "+
               "The GO category can be selected by clicking on its row in the list.  "+ 
               "The p-value GO category enrichment for K-means clusters are computed based on the "+
               "cluster's actual size. \n"+
               "*  User Defined Gene Set Enrichment -- this is similar to GO Category enrichment "+
               "except that the set of genes can be arbitrarily defined.\n"+
               "*  Cluster ID -- reorders K-means clusters by their ID number, that is the number in the top "+
               "left hand corner of the cluster  box.\n" +
               "*  Number of Genes -- reorders profiles based on the number of genes assigned to the profile; also "+
                "displays this value in the bottom left hand corner of the cluster box.\n\n" +
               "Note: "
	       +"The table can be sorted by any of the columns by clicking on the column's header.  "
               +"Using the 'Copy Table' the entire table can be copied to the clipboard.\n"+
	       "Using the 'Save Table' button the table can be saved to a file.";
	      Util.renderDialog(theFrame,szMessage,-90,-15);
	   }
	   else if (theDataSet.tga.bcluster)
           {
              szMessage =
               "Clusters, similar significant profiles that are colored the same, are ordered on the screen from top "+
               "to bottom and left to right.  " +
	       "By default clusters are ordered by number of genes assigned.  Non-significant profiles, which "+
               "are not clustered come last.  Clusters are ordered by size.  Within "+
                "each cluster, and among the non-significant profiles, the profiles are ordered in "+
                "increasing order of the p-value of the "+
                "number of genes assigned to the profile as compared to the number expected based on a "+
		  "permutation test.\n\n"+
               "Clusters can be reordered from top to bottom and "+
	       "left to right by one of several criteria: \n"+
               "*  GO Category Gene Enrichment -- the clusters can be reordered by "+
               "gene enrichment of genes annotated as belonging to any GO Category.  "+
               "The GO category can be selected by clicking on its row in the list.  "+ 
               "The p-value GO category enrichment for profiles can be computed based on only "+
		  "the cluster's actual size, unlike profiles which are defined independent of the data "+
                "clusters do not have an expected size.  "+
               "The table contains for each GO Category "+
               "the minimum p-value enrichment for any profile.\n"+
               "*  User Defined Gene Set Enrichment -- this is similar to GO Category enrichment "+
               "except that the set of genes can be arbitrarily defined.\n"+
	       "*  Default Order --  reorders the profiles to their "+
               "original order.\n\n"
               +"Note: "
	       +"The table can be sorted by any of the columns by clicking on the column's header.  "
               +"Using the 'Copy Table' the entire table can be copied to the clipboard.\n"+
	       "Using the 'Save Table' button the table can be saved to a file.";
	      Util.renderDialog(theFrame,szMessage,-90,-15);
	   }
	   else
	   {
              szMessage =
                "Profiles are ordered on the screen from top to bottom and left to right.  " +
		  "By default profiles of the same cluster are placed next to each other, with "+
                "the non-significant profiles coming last.  Within "+
                "each cluster, and among the non-significant profiles, the profiles are ordered in "+
                "increasing order of the p-value of the "+
                "number of genes assigned to the profile as compared to the number expected based on a "+
		  "permutation test.\n\n"+
               "Profiles can be reordered from top to bottom and "+
	       "left to right by one of several criteria: \n"+
               "*  GO Category Gene Enrichment -- the profiles can be reordered by "+
               "gene enrichment of genes annotated as belonging to any GO Category.  "+
               "The GO category can be selected by clicking on its row in the list.  "+ 
               "The p-value GO category enrichment for profiles can be computed based on either "+
               "a profile's actual or expected size.  "+
	       "For instance consider a profile with many more genes assigned than expected, but given "+
               "the number of genes assigned to profile the number of genes that belong to the profile and the "+
	       "GO Category is what is expected.  "+
               "In this case the profile will be enriched by "+
               "computing p-values based on the profile's expected size, but not based on its actual size.  "+
               "The table contains for each GO Category and both methods of computing p-values "+
               "the minimum p-value enrichment for any profile.\n"+
               "*  User Defined Gene Set Enrichment -- this is similar to GO Category enrichment "+
               "except that the set of genes can be arbitrarily defined.\n"+
               "*  Profile ID -- reorders profiles by their ID number, that is the number in the top "+
               "left hand corner of the profile  box.\n" +
               "*  Significance -- reorders profiles by the p-value significance of the number of genes "+
               "assigned to the profile compared to what was expected; also displays this p-value in the bottom  "+
               "left-hand corner of each profile box.\n" +
               "*  Number of Genes -- reorders profiles based on the number of genes assigned to the profile; also "+
                "displays this value in the bottom left hand corner of the profile box.\n" +
               "*  Expected Number -- reorders profiles based on the expected number of genes assigned to "+
                "this profile based on a permutation test; "+
               "also displays this value in the bottom left hand corner of the profile box.\n"+
	       "*  Default Order --  reorders the profiles to their "+
               "original order.\n\n"
               +"Note: "
	       +"+The table can be sorted by any of the columns by clicking on the column's header.\n"
               +"+Using the 'Copy Table' the entire table can be copied to the clipboard.\n"+
	       "+Using the 'Save Table' button the table can be saved to a file.";
	      Util.renderDialog(theFrame,szMessage,-200,-100);
	   }	   
        }
     }
}

