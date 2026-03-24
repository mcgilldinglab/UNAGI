package edu.cmu.cs.sb.chromviewer;

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
import java.io.*;
import javax.swing.*;
import javax.swing.table.*;
import java.awt.datatransfer.*;

/**
 * Table of the genes being displayed on the chromosome viewer
 */
public class ChromGeneAcceptTable extends JPanel implements ActionListener 
{
    private boolean DEBUG = false;

    ChromFrame cf;
    String[] columnNames;
    String[][] tabledata;
    JButton saveButton,copyButton;
    JButton savenamesButton,copynamesButton;
    ArrayList alprofiles;
    TableSorter sorter;
    final static Color bg = Color.white;
    final static Color fg = Color.black;

    boolean bprofileonly;
    boolean bquery;
    String szTitle;
    
    JFrame theFrame;

    /**
     * Constructor - renders the table
     */
    public ChromGeneAcceptTable(JFrame theFrame,HashMap hmaccepted,
				Gene[] chromNames, String szGeneHeader)
    {
        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

        setBackground(bg);
        setForeground(fg);

	this.theFrame = theFrame;

        int numcols = 5;
        columnNames = new String[numcols];
        columnNames[0] = szGeneHeader;
	columnNames[1] = "Chromosome";
	columnNames[2] = "Strand";
	columnNames[3] = "Begin";
	columnNames[4] = "End";
        int ngeneindex = 0;

        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);

	int numuniquegenes = hmaccepted.size();
	Iterator hmacceptedels = hmaccepted.values().iterator();
	int ntabledatasize = 0;
	for (int ngene = 0; ngene < numuniquegenes; ngene++)
	{
	    HashSet currhs = (HashSet) hmacceptedels.next();
	    ntabledatasize += currhs.size();
	}

        tabledata = new String[ntabledatasize][columnNames.length];
	Iterator  hmaccepteditr= hmaccepted.entrySet().iterator();       
        int ntableindex = 0;
	for (int ngene = 0; ngene < numuniquegenes; ngene++)
	{
	   Map.Entry theEntry = (Map.Entry) hmaccepteditr.next();
	   String szgene = (String) theEntry.getKey();
	   HashSet currhs = (HashSet)theEntry.getValue();
	   Iterator currhsitr = currhs.iterator();
	   while (currhsitr.hasNext())
	   {
	      Gene theGene = (Gene) currhsitr.next();
              tabledata[ntableindex][0] = szgene;
              tabledata[ntableindex][1] = chromNames[theGene.chromosome].namechrom;
              tabledata[ntableindex][2] = ""+theGene.strand;
              tabledata[ntableindex][3] = ""+theGene.start;
              tabledata[ntableindex][4] = ""+theGene.end;
	      ntableindex++;
	   }
	}

        sorter = new TableSorterRoman(new TableModelST(tabledata,columnNames),1);
        final JTable table = new JTable(sorter);

        sorter.setTableHeader(table.getTableHeader());
        table.setPreferredScrollableViewportSize(new Dimension(300, 
                             Math.min((table.getRowHeight()+table.getRowMargin())*
				      table.getRowCount(),400)));

        TableColumn column;
        column = table.getColumnModel().getColumn(0);
        column.setPreferredWidth(100);
        column = table.getColumnModel().getColumn(1);
        column.setPreferredWidth(75);
        column = table.getColumnModel().getColumn(2);
        column.setPreferredWidth(40);
        column = table.getColumnModel().getColumn(3);
        column.setPreferredWidth(100);
        column = table.getColumnModel().getColumn(3);
        column.setPreferredWidth(100);

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

        JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpButton.addActionListener(this);
        helpButton.setActionCommand("help");
        buttonPanel.add(helpButton);
	buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE,20));
        add(buttonPanel);
    }

    /**
     *Outputs the content of the PrintWriter
     */
    public void printFile(PrintWriter pw)
    {
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
     * Copies the contents of the table to the clipboard
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
     * Copies the gene names to the clipboard 
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
     * Writes the gene names to a file
     */
    public void printGeneList(PrintWriter pw)
    {
       for (int nrow = 0; nrow <tabledata.length; nrow++)
       {
	  pw.println(sorter.getValueAt(nrow,2));
       }
    }

    /**
     * Handles actions on the interface
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
          String szMessage = "The table contains genes currently displayed in the chromsome viewer "+
                         "and their locations including chromosome, strand, and beginning and end position. ";
	  Util.renderDialog(theFrame,szMessage);
       }
    }
}

