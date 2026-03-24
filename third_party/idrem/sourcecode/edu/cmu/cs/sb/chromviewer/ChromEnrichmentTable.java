package edu.cmu.cs.sb.chromviewer;

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
 * Table for an enrichment analysis on the set of genes assigned to each chromosome
 */ 
public class ChromEnrichmentTable extends JPanel implements ActionListener
{
    private boolean DEBUG = false;


    JFrame theFrame;
    DataSetCore theDataSet;
    GoAnnotations.GoResults tgr;

    final static Color bg = Color.white;
    final static Color fg = Color.black;
    JButton saveButton,copyButton;

    String[][] tabledata= null;
    String[] columnNames = null;
    TableSorterRoman sorter;
    HashSet htinames;
    String szTitle;
    ChromFrame cf;


    final static int PVALUE_INDEX = 5;
    final static int CORRECTEDPVALUE_INDEX = 6;

    /**
     * Constructor - renders the enrichment table
     */
    public ChromEnrichmentTable(JFrame theFrame,GenomeFileParser genomeParser,int nactiveunique) 
    {
        //super(new GridLayout(0,1));
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
        setBackground(bg);
        setForeground(fg);
        this.theFrame = theFrame;

        NumberFormat nf1 =  NumberFormat.getInstance(Locale.ENGLISH);
        nf1.setMinimumFractionDigits(1);
        nf1.setMaximumFractionDigits(1);
        nf1.setGroupingUsed(false);

        int numcols = 7;
	Gene[] m_chromArray = genomeParser.getChromArray();
        columnNames = new String[numcols];
	
        columnNames[0] = "Chromosome";
        columnNames[1] = "#Genes\nBase Set";
        columnNames[2] = "#Genes\nAssigned";
        columnNames[3] = "#Genes\nExpected";
        columnNames[4] = "#Genes\nEnriched";
        columnNames[PVALUE_INDEX] = "p-value";
        columnNames[CORRECTEDPVALUE_INDEX] = "Corrected\np-value";

        tabledata = new String[m_chromArray.length][];

        NumberFormat nf3 = NumberFormat.getInstance(Locale.ENGLISH);
        nf3.setMinimumFractionDigits(3);
        nf3.setMaximumFractionDigits(3);
        
	int[] baseDist = genomeParser.getBaseCountPerChrom();
	int[] activeDist = genomeParser.getActiveCountPerChrom();

	int baseTotal = 0;
	int activeTotal = 0;

	for (int i = 0; i < m_chromArray.length; i++) 
        {
	    baseTotal += baseDist[i];
	    activeTotal += activeDist[i];
	}

	RecRow[] sortedRecs = new RecRow[m_chromArray.length];
        for (int nindex = 0; nindex < m_chromArray.length; nindex++)
	{
	   String[] therow = new String[columnNames.length];
	   therow[0] = m_chromArray[nindex].namechrom;
	   therow[1] = ""+baseDist[nindex];
           therow[2] = ""+ activeDist[nindex];
	   double expectedvalue = baseDist[nindex] * activeTotal /(double) baseTotal;
	   therow[3] = nf1.format(expectedvalue);
	   double diff = activeDist[nindex] - expectedvalue;
	   therow[4] = nf1.format(diff);
	 
	   double dpval = StatUtil.hypergeometrictail(
                 activeDist[nindex]-1,baseDist[nindex],baseTotal-baseDist[nindex],activeTotal);
	   double dcorrectedpval = m_chromArray.length * dpval;
	   therow[PVALUE_INDEX] = Util.doubleToSz(dpval);

	   if (!genomeParser.brandcorrectedpval)
	   {
	      therow[CORRECTEDPVALUE_INDEX] = Util.doubleToSz(Math.min(1,dpval*m_chromArray.length));
	   }

           double dval = Double.parseDouble(therow[3]);
           if (dval > 0)
           {
              therow[3] = "+" + therow[3];
           }
           else if (dval == 0)
           {
	      therow[3] = "0.0";
           }
	   sortedRecs[nindex] = new RecRow(therow,dpval);
	}

	Arrays.sort(sortedRecs,new RecRowCompare());

	if (genomeParser.brandcorrectedpval)
	{	    
            ComputeCorrectedPval(sortedRecs,genomeParser, baseDist,
                                genomeParser.nsamplespval,nactiveunique);
	}

        for(int nindex = 0; nindex < sortedRecs.length; nindex++)
	{
	    tabledata[nindex] = sortedRecs[nindex].therow;
	}

        sorter = new TableSorterRoman(new TableModelST(tabledata, columnNames),0);
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
         column.setPreferredWidth(40);
         column = table.getColumnModel().getColumn(1);
         column.setPreferredWidth(40);
         column = table.getColumnModel().getColumn(2);
         column.setPreferredWidth(40);
         column = table.getColumnModel().getColumn(3);
         column.setPreferredWidth(40);
         column = table.getColumnModel().getColumn(4);
         column.setPreferredWidth(40);
         column = table.getColumnModel().getColumn(5);
         column.setPreferredWidth(40);
           
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
     * Writes the content of the table to a file
     */
    public void printFile(PrintWriter pw)
    {      
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
     * Copies the content of the table to a clipboard
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
     * Handles interface actions
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
           String szMessage = "The table contains a Chromosome enrichment analysis "+
                              "for the set of genes in the displayed set. The columns are as follows "+
                              "+'Chromosome' - the ID of the Chromosome\n"+
                              "+'#Genes Base Set' - how many genes were assigned to that chromosome in the "+
                              "base set of genes before any filtering\n"+
                              "+'#Genes Assigned' - the # genes in the current set assigned to that chromosome\n"+
                              "+'#Genes Expected' - the # genes expected to be assigned to chromosome if a set of the same size as currently "+
                              "displayed was randomly drawn from the base set without replacement\n"+
                              "+'#Genes Enriched' - the difference between the number of genes assigned and expected\n"+
                              "+'p-value' - the enrichment p-value based on the hypergeometric distribution\n"+
                              "+'Corrected p-value' - the corrected p-value for the enrichment using the "+
                              "same method as specified for the p-value correction method for "+
	                      "the Go Analysis (either Bonferroni or Randomization).";

	   Util.renderDialog(theFrame,szMessage);
       }
    }

    /**
     * Record containing a p-value and the name of the chromosome
     */
    static class RecRow
    {
	RecRow(String[] therow,double dpval)
	{
	    this.therow = therow;
	    this.dpval = dpval;
	}
	String[] therow;
	double dpval;
    }

    /**
     * Sorts first based on p-value, then based on chromosome
     */
    static class RecRowCompare implements Comparator
    {
       public int compare(Object o1, Object o2)
       {
	  RecRow d1 = (RecRow) o1;
	  RecRow d2 = (RecRow) o2;

	  if (d1.dpval < d2.dpval)
	  {
	      return -1;
	  }
	  else if (d1.dpval > d2.dpval)
	  {
	      return 1;
	  }
	  else
	  {
	      return d1.therow[0].compareTo(d2.therow[0]);
	  }
       }
    }

    /**
     * Computes corrected p-values for chromosome enrichments
     */     
    private void ComputeCorrectedPval(RecRow[] theRecRow,GenomeFileParser genomeParser,int[] baseDist,  
                                      int nsamplespval,int nrandom)
    {
       HashMap htHyper = new HashMap();
       int nbaseselect = genomeParser.basegenesets.length;
 
       //randomized multiple hypothesis correction
       //nrandom is the number of samples to draw
       int[] rindex = new int[nrandom];
       //rindex contains the random selected indicies

       //nsamplespval is the number of trials
       double[] dminpval = new double[nsamplespval];
       int[] ncount = new int[nrandom];

       int[] nummatches = new int[genomeParser.m_chromArray.length];
       //ncount contains the smallest category size for that gene set size

       int nmaxsize = 0;
       for (int nindex = 0; nindex < baseDist.length; nindex++)
       {
	   if (baseDist[nindex]> nmaxsize)
	   {
	       nmaxsize = baseDist[nindex];
	   }
       }

       for (int nsample = 0; nsample < nsamplespval; nsample++)
       {
          for (int nindex = 0; nindex < ncount.length; nindex++)
          {
             ncount[nindex] = nmaxsize+1;
	  }

	  dminpval[nsample] = 1;
          
          for (int nindex = 0; nindex < nrandom; nindex++)
          {
	     rindex[nindex] = nindex;
	  }

          //drawing nrandom elements from a set of numtotalgenes elements
          //where each element is equally likely
          for (int nindex = nrandom; nindex < nbaseselect; nindex++)
          {
	     if (Math.random() < ((double) nrandom/(double) (nindex+1)))
	     {
	        rindex[(int) Math.floor(nrandom*Math.random())] = nindex;
	     }
          }
	     
          //random genes selected now going to score them
          HashSet hsChrom;
	  for (int nindex = 0; nindex < nummatches.length; nindex++)
	  {
	      nummatches[nindex] = 0;
	  }

	  for (int nindex = 0; nindex < nrandom; nindex++) 
          {        
	     //going through each gene getting its GO results
             hsChrom = genomeParser.basegenesets[rindex[nindex]];
	     //System.out.println(hsChrom);
	     Iterator hsChromitr = hsChrom.iterator();
	     while (hsChromitr.hasNext())
	     {
		int nchrom = ((Gene) hsChromitr.next()).chromosome;
		nummatches[nchrom]++;
	     }
	  }

	  for (int nindex  = 0; nindex < nummatches.length; nindex++)
	  {
	      int ncategoryselect = nummatches[nindex];
	      int ncategoryall = baseDist[nindex];
              
              if ((ncategoryselect >=1)&&(ncount[--ncategoryselect] > ncategoryall))
              {
	         //found a smaller category with the same number selected
                 ncount[ncategoryselect] = ncategoryall;
	      }
	  }

	  //trying to find the smallest p-value
          int nsmallest = nmaxsize+1;
          for (int nindex = ncount.length-1; nindex >= 0; nindex--)
          {
             if (ncount[nindex] < nsmallest)
             {
	        //to have a better p-value category size must be smaller than for a larger
	        //number of genes selected
                String szpair = nindex+","+ncount[nindex];
                Object pvalObj = htHyper.get(szpair);
                double dpval;
                nsmallest = ncount[nindex];
                if (pvalObj == null)
                {
                   dpval = StatUtil.hypergeometrictail(nindex,ncount[nindex],  
                                               nbaseselect-ncount[nindex], nrandom);
                   htHyper.put(szpair, new Double(dpval));
                }
                else
                {
                   dpval = ((Double) pvalObj).doubleValue(); 
                }

  	        if (dpval < dminpval[nsample])
                {
	           dminpval[nsample] = dpval;
                }
	     }       
	  }
       }
       Arrays.sort(dminpval);

       int npvalindex = 0;
       for (int nrecindex = 0; nrecindex < theRecRow.length; nrecindex++)
       {
	  double dcurrpval = Double.parseDouble(theRecRow[nrecindex].therow[PVALUE_INDEX]);
          while ((npvalindex<dminpval.length)&&
		 (dminpval[npvalindex] <= dcurrpval))
          {
             npvalindex++;
	  }
          theRecRow[nrecindex].therow[CORRECTEDPVALUE_INDEX] =""+ (npvalindex/(double) dminpval.length);
       }           
    }
    
}

