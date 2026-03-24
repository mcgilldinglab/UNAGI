package edu.cmu.cs.sb.stem;


import edu.cmu.cs.sb.core.*;
import javax.swing.*;
import javax.swing.ImageIcon;
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


/**
 * Class encapsulates the window slecting which genes to select and unselect
 */
public class DefineGeneSet extends JPanel implements ActionListener
{
    /**
     * Background color
     */
    final static Color bg = Color.white;

    /**
     * Foreground color
     */
    final static Color fg = Color.black;

    /**
     * The GoAnnotations object
     */
    GoAnnotations tga;

    /**
     * The table model for this table
     */
    TableModelSetST tms;

    /**
     * The maingui that this geneset connects with
     */
    MAINGUI2 maingui;

    /**
     * The frame in which this panel will be added 
     */
    JFrame theFrame;

    /**
     * The limit on the number of not found genes when loading a file
     */
    static int NMAXBAD = 30;

    /**
     * Controls sorting in the table
     */
    TableSorter sorter;

    /**
     * Controls selecting and unselecting rows
     */
    ListSelectionModel rowSM;

    /**
     * To select the checked genes 
     */
    JButton selectgenesButton;

    /**
     * Whether the clustering method is the STEM profile metho or kmeans clustering
     */
    boolean bkmeans;

    /**
     * Spinner for selecting profile in comparison experiment
     */
    JSpinner thespinnerprofiles;

    /**
     * Constructor for class. Initializes variables and makes interface window
     */
    public DefineGeneSet(JFrame theFrame, MAINGUI2 maingui, GoAnnotations tga, 
                         ListSelectionModel rowSM,boolean bkmeans) 
    {
        this.theFrame = theFrame;
        this.tga = tga;
        this.maingui = maingui;
	this.rowSM = rowSM;
	this.bkmeans = bkmeans;
        makewindow();
    }


    /**
     * Sets all genes to the value of bval
     */
    void setAll(boolean bval)
    {
       for (int nrow = 0; nrow < tms.data.length; nrow++)
       {
          tms.setValueAt(Boolean.valueOf(bval), nrow, 1);
       }
    } 

    /**
     * Selects all genes assigned to nprofile in the other data set. 
     * If -1 loads the filtered genes.
     */
    void loadProfile(int nprofile)
    {                  
       HashMap htLoadGenes = new HashMap();

       if (nprofile >= 0)
       {
          ArrayList genelist = maingui.theDataSet.otherset.profilesAssigned[nprofile];
          int numgenes = genelist.size();
          for (int ngeneindex = 0; ngeneindex < numgenes; ngeneindex++)
          {
	     int nindex = ((Integer) genelist.get(ngeneindex)).intValue();
             htLoadGenes.put(maingui.theDataSet.otherset.genenames[nindex], Boolean.valueOf(true));
          }
       }
       else
       {
	   Iterator itr = maingui.theDataSet.otherset.htFiltered.keySet().iterator();
	   while (itr.hasNext())
	   {
	       htLoadGenes.put(itr.next(),Boolean.valueOf(true));
	   }
       }

       for (int nrow = 0; nrow < tms.data.length; nrow++)
       {
          if (htLoadGenes.get(tms.data[nrow][0]) != null)
	  {
	      tms.setValueAt(Boolean.valueOf(true), nrow, 1);
	  }
       }             
       sorter.setSortingStatus(1, TableSorter.DESCENDING); 
    }

    /**
     * Controls actions for various buttons being pressed on the define gene set interface
     */
    public void actionPerformed(ActionEvent e) 
    {
       String szcommand = e.getActionCommand();
       int nbadline = 0;

       if (szcommand.equals("selectgenes"))
       {
           int nprofile = ((Integer) thespinnerprofiles.getValue()).intValue();
           loadProfile(nprofile);
       }
       else if (szcommand.equals("selectall"))
       {
	   setAll(true);
       }
       else if (szcommand.equals("selectnone"))
       {
	   setAll(false);
       }
       else if (szcommand.equals("query"))
       {
          if ((tga.bactual)||(tga.bcluster))
          {
             tga.szsortcommand = "define";
          }
          else
          {
             tga.szsortcommand = "expdefine";
          }
          rowSM.clearSelection();
          tga.szSelectedGO = null; 
          maingui.drawmain();
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
              ex.printStackTrace(System.out);
          }
       }
       else if (szcommand.equals("load"))
       {

          setAll(false);

          int returnVal = ST.fc.showOpenDialog(this);
	  String szMissing = "";
          StringBuffer szMissingBuf = new StringBuffer();

          if (returnVal == JFileChooser.APPROVE_OPTION) 
          {
	     BufferedReader br = null;
             try
	     {
                File file = ST.fc.getSelectedFile();
                FileReader fr = new FileReader(file);
                br = new BufferedReader(fr);
                String szLine;
                HashMap htLoadGenes = new HashMap();
                HashMap htMissing = new HashMap();
   
                while ((szLine = br.readLine())!=null)
	        {
		   String szname = szLine.trim().toUpperCase(Locale.ENGLISH);
                   if (tga.htGeneNames.get(szname)!= null)
		   {
		       htLoadGenes.put(szname, Boolean.valueOf(true));
		   }
                   else if ((!szname.equals("0"))&&(htMissing.get(szname)==null)&&(nbadline<NMAXBAD))
		   {
		      szMissingBuf.append(szLine+"\n");
                      htMissing.put(szname, Boolean.valueOf(true));
                      nbadline++;
		   }
		}

		szMissing = szMissingBuf.toString();

                for (int nrow = 0; nrow < tms.data.length; nrow++)
		{
		    if (htLoadGenes.get(tms.data[nrow][0]) != null)
		    {
			tms.setValueAt(Boolean.valueOf(true), nrow, 1);
		    }
                }
	     }
             catch (FileNotFoundException ex)
	     {
                JOptionPane.showMessageDialog(this, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                ex.printStackTrace(System.out);
             }
             catch (IOException ex)
	     {
                JOptionPane.showMessageDialog(this, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                ex.printStackTrace(System.out);
             }
	     finally
	     {
		 if (br != null)
		 {
                     try
		     {
		        br.close();
		     }
		     catch (IOException ex)
		     {
                        JOptionPane.showMessageDialog(this, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                        ex.printStackTrace(System.out);
		     }   
		 }
	     }

             //want to show genes loaded
             sorter.setSortingStatus(1, TableSorter.DESCENDING);    
          }  
 
          if (!szMissing.equals(""))
	  {
              if (nbadline < NMAXBAD)
	      {
	         szMissing = "The following genes from the loaded gene "+
		  "set are not in the data file:\n"+szMissing;
	      }
              else
	      {
	         szMissing = "The following are the first "+NMAXBAD+" genes from the loaded gene "+
		  "set that are not in the data file:\n"+szMissing;
	      }
               JOptionPane.showMessageDialog(this, szMissing, 
                               "Genes Not in the Data File", JOptionPane.WARNING_MESSAGE);

	  }
       }
       else if (szcommand.equals("help"))
       {
           String szMessage;


	   if (maingui.theDataSet.bkmeans)
	   {
              szMessage =
                 "This window allows one to select a set of genes to query.  When the 'Query Gene Set' "+
              "button is pressed K-means clusters are reordered based on enrichment of the query set of genes. "+
              "Also the number of genes from the query set assigned to each cluster and the p-value of the "+
              "enrichment will appear in "+
	      "the lower left hand corner of the cluster boxes.\n\n"+
              "The genes that form a query set can be specified in one of several ways:\n"+
              "*The genes that define a query set can be selected or unselected by manually checking "+
              "or unchecking the boxes next to their name in the list.\n"+
              "*All the genes can be selected with the 'Select All' button.\n"+
              "*All the genes can be set to unselected with the 'Unselect All' button.\n"+
	      "*A gene set can be loaded from a file through the 'Load Gene Set' option.  "+
              "Each gene to be included in the query set should appear in a file one per line.\n"+
	      "*If a comparison set has been specified through the 'Compare...' option on the main window, "+ 
              "then the genes in the comparison set assigned to one or more specific clusters can be selected "+
              "by choosing the cluster ID and then pressing 'Select Genes'.  Likewise if the current set "+
              "of clusters is from a comparison experiment, the set of genes assigned to cluster(s) in the original "+
              "experiment can be loaded in the same manner.  "+
              "To select all the genes filtered in the other experiment "
              + "set the cluster ID to -1 and then press 'Select Genes'.\n\n"+
              "Note:\n"+
	      "+The table can be sorted by any of the columns by clicking on the column's header.\n"+
	      "+Using the 'Save Gene Set' button the set of selected genes can be saved.  A saved gene list "+
	      "can then be loaded at a later time through the 'Load Gene Set' option.\n";
	   }
	   else
	   {
              szMessage =
              "This window allows one to select a set of genes to query.  When the 'Query Gene Set' "+
              "button is pressed profiles are reordered based on enrichment of the query set of genes. "+
              "Also the number of genes from the query set assigned to each profile and the p-value of the "+
              "enrichment will appear in "+
	      "the lower left hand corner of the profile boxes.\n\n"+
              "The genes that form a query set can be specified in one of several ways:\n"+
              "*The genes that define a query set can be selected or unselected by manually checking "+
              "or unchecking the boxes next to their name in the list.\n"+
              "*All the genes can be selected with the 'Select All' button.\n"+
              "*All the genes can be set to unselected with the 'Unselect All' button.\n"+
	      "*A gene set can be loaded from a file through the 'Load Gene Set' option.  "+
              "Each gene to be included in the query set should appear in a file one per line.\n"+
	      "*If a comparison set has been specified through the 'Compare...' option on the main window, "+ 
              "then the genes in the comparison set assigned to one or more specific profiles can be selected "+
              "by choosing the profile ID and then pressing 'Select Genes'.  Likewise if the current set "+
              "of profiles is from a comparison experiment, the set of genes assigned to profile(s) in the original "+
              "experiment can be loaded in the same manner.  "+
              "To select all the genes filtered in the other experiment "
              + "set the profile ID to -1 and then press 'Select Genes'.\n\n"+
              "Note:\n"+
	      "+The table can be sorted by any of the columns by clicking on the column's header.\n"+
	      "+Using the 'Save Gene Set' button the set of selected genes can be saved.  A saved gene list "+
	      "can then be loaded at a later time through the 'Load Gene Set' option.\n";
	   }

	   Util.renderDialog(theFrame,szMessage,-200,-100);
       }
    }

    /**
     * Prints those genes currently selected out to a file
     */
    public void printFile(PrintWriter pw)
    {
        for (int nrow = 0; nrow < tms.data.length; nrow++)
	{
	    if (((Boolean) tms.data[nrow][1]).booleanValue())
	    {
		pw.println(tms.data[nrow][0]);
	    }

	}
    }


    /**
     * Controls the interface window for defining gene sets
     */
    public void makewindow()
    {
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
        setBackground(bg);
        setForeground(fg);

        Iterator eNames = tga.htGeneNames.keySet().iterator();

        int nsize = tga.htGeneNames.size();
        Object[][] tabledata = new Object[nsize][2];
        String[] columnNames = new String[2];
	
        columnNames[0] = "Gene Name";
        columnNames[1] = "In Gene Set";
        String[] fullgenenames = new String[nsize];

        for (int nrow = 0; nrow < nsize; nrow++)
	{
            fullgenenames[nrow] = (String) eNames.next();
	}
        Arrays.sort(fullgenenames);

        boolean btrue = false;
        for (int nrow = 0; nrow < nsize; nrow++)
	{    
            tabledata[nrow][0] = fullgenenames[nrow];
            tabledata[nrow][1] = tga.htGeneNames.get(fullgenenames[nrow]);
            btrue = (btrue || ((Boolean)tabledata[nrow][1]).booleanValue());
	}
	
        tms = new TableModelSetST(tabledata,columnNames,tga);
       
        sorter = new TableSorter(tms);
        if (btrue)
	{
           sorter.setSortingStatus(1, TableSorter.DESCENDING);
	} 
        final JTable table = new JTable(sorter);
        sorter.setTableHeader(table.getTableHeader());
           
        TableColumn column;
        table.setPreferredScrollableViewportSize(new Dimension(425, 300));
        column = table.getColumnModel().getColumn(0);
        column.setPreferredWidth(200);
        column = table.getColumnModel().getColumn(1);
        column.setPreferredWidth(50);
        //Create the scroll pane and add the table to it.
        JScrollPane scrollPane = new JScrollPane(table);

        //Add the scroll pane to this panel.
        add(scrollPane);

	JLabel compareLabel;

	String szprefix;
	if (bkmeans)
	{
	    szprefix = "Cluster";
	}
	else
	{
	    szprefix = "Profile";
	}

        if ((maingui.theDataSet.otherset!=null)&&
	    (maingui.theDataSet.bothersetorigset))
	 {
             compareLabel = new JLabel(szprefix+" ID in Original Set: ");
	 }
	else
	{
            compareLabel = new JLabel(szprefix+" ID in Comparison Set: ");
	}

	JPanel comparePanel = new JPanel();
        comparePanel.setBackground(Color.white);
        int nmid = maingui.theDataSet.modelprofiles.length/2; 
        SpinnerNumberModel snprofiles  = new SpinnerNumberModel(new Integer(nmid),new Integer(-1), 
				   new Integer(maingui.theDataSet.modelprofiles.length-1), new Integer(1));
	//changed to -1 to allow genes filtered in the other data set to be selected
        thespinnerprofiles = new JSpinner(snprofiles);   
        selectgenesButton = new JButton("Select Genes");

	if (maingui.theDataSet.otherset == null)
	{
	   selectgenesButton.setEnabled(false);  
	}   
        comparePanel.add(compareLabel);
        comparePanel.add(thespinnerprofiles);
        selectgenesButton.setActionCommand("selectgenes");
        selectgenesButton.addActionListener(this);
        comparePanel.add(selectgenesButton);
	comparePanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
	add(comparePanel);

        JPanel buttonPanel = new JPanel();
        buttonPanel.setBackground(Color.white);

        JButton selectnoneButton = new JButton("Unselect All");
        selectnoneButton.setActionCommand("selectnone");
        selectnoneButton.addActionListener(this);
        buttonPanel.add(selectnoneButton); 

        JButton selectallButton = new JButton("Select All");
        selectallButton.setActionCommand("selectall");
        selectallButton.addActionListener(this);
        buttonPanel.add(selectallButton); 
	buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel);
        JPanel buttonPanel2 = new JPanel();
        buttonPanel2.setBackground(Color.white);

        JButton queryButton = new JButton("Query Gene Set");
        queryButton.setActionCommand("query");
        queryButton.addActionListener(this);
        buttonPanel2.add(queryButton);

        JButton loadButton = new JButton("Load Gene Set",Util.createImageIcon("Open16.gif"));
        loadButton.setActionCommand("load");
        loadButton.addActionListener(this);
        buttonPanel2.add(loadButton);  

        JButton saveButton = new JButton("Save Gene Set",Util.createImageIcon("Save16.gif"));
        saveButton.setActionCommand("save");
        saveButton.addActionListener(this);
        buttonPanel2.add(saveButton);  
    
        JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpButton.addActionListener(this);
        helpButton.setActionCommand("help");
        buttonPanel2.add(helpButton);
	buttonPanel2.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel2);
    }
}
