package edu.cmu.cs.sb.drem;

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
 * Class for a table summarizing the TF-gene enrichments for a selected set of
 * gene This is the table that comes up when pressing the TF summary button
 */
public class DREMGui_StaticSummaryTable extends JPanel implements
		ActionListener {
	DREM_DataSet theDataSet;
	String[] columnNames;
	String[][] tabledata;
	double[][] bindingpvalTF;
	int[][] bindingpvalTFindex;
	double[] davg;
	double[] dstd;
	String[] tfNames;
	JButton saveButton;
	JButton copyButton;
	JButton staticButton;
	TableSorter sorter;
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	JFrame theFrame;
	boolean[] include;
	int numgenes;
	DREM_Timeiohmm theTimeiohmm;

	/**
	 * Constructor builds the table
	 */
	public DREMGui_StaticSummaryTable(JFrame theFrame, DREM_DataSet theDataSet,
			DREM_Timeiohmm theTimeiohmm, double[][] bindingpvalTF,
			int[][] bindingpvalTFindex, String[] tfNames, boolean[] include,
			int numgenes) {
		this.theFrame = theFrame;
		this.theDataSet = theDataSet;
		this.include = include;
		this.bindingpvalTF = bindingpvalTF;
		this.bindingpvalTFindex = bindingpvalTFindex;
		this.tfNames = tfNames;
		this.theTimeiohmm = theTimeiohmm;
		this.numgenes = numgenes;
		geneTableHelper();
	}

	/**
	 * Outputs the content of the PrintWriter
	 */
	public void printFile(PrintWriter pw) {

		for (int ncol = 0; ncol < columnNames.length - 1; ncol++) {
			pw.print(columnNames[ncol] + "\t");
		}
		pw.println(columnNames[columnNames.length - 1]);

		for (int nrow = 0; nrow < tabledata.length; nrow++) {
			for (int ncol = 0; ncol < tabledata[nrow].length - 1; ncol++) {
				pw.print(sorter.getValueAt(nrow, ncol) + "\t");
			}
			pw.println(sorter.getValueAt(nrow, columnNames.length - 1));
		}
	}

	/**
	 * Copies the content of the table to the clipboard
	 */
	public void writeToClipboard() {
		StringBuffer sbuf = new StringBuffer();

		// get the system clipboard
		Clipboard systemClipboard = Toolkit.getDefaultToolkit()
				.getSystemClipboard();

		for (int ncol = 0; ncol < columnNames.length - 1; ncol++) {
			sbuf.append(columnNames[ncol] + "\t");
		}
		sbuf.append(columnNames[columnNames.length - 1] + "\n");

		for (int nrow = 0; nrow < tabledata.length; nrow++) {
			for (int ncol = 0; ncol < tabledata[nrow].length - 1; ncol++) {
				sbuf.append(sorter.getValueAt(nrow, ncol) + "\t");
			}
			sbuf.append(sorter.getValueAt(nrow, columnNames.length - 1) + "\n");
		}
		// set the textual content on the clipboard to our
		// Transferable object
		Transferable transferableText = new StringSelection(sbuf.toString());
		systemClipboard.setContents(transferableText, null);
	}

	/**
	 * Builds the table
	 */
	private void geneTableHelper() {
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));

		setBackground(bgColor);
		setForeground(fgColor);

		int numcols = 6;
		columnNames = new String[numcols];
		columnNames[0] = "TF";
		columnNames[1] = "Total Overall";
		columnNames[2] = "Selected";
		columnNames[3] = "Expected Overall";
		columnNames[4] = "Diff Overall";
		columnNames[5] = "Overall Score";

		NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);

		int numtf = theTimeiohmm.bindingData.regNames.length;
		int nsize = theTimeiohmm.bindingData.existingBindingValuesUnsorted
				.size();
		if (theTimeiohmm.bindingData.existingBindingValuesUnsorted
				.contains(new Double(0.0))) {
			nsize--;
		}
		int numrows = numtf * nsize;

		tabledata = new String[numrows][numcols];
		int nrowindex = 0;
		for (int nrow = 0; nrow < numtf; nrow++) {
			for (int nel = 0; nel < theTimeiohmm.bindingData.existingBindingValuesSorted.length; nel++) {
				if (theTimeiohmm.bindingData.existingBindingValuesSorted[nel] != 0) {
					int ncount = 0;
					tabledata[nrowindex][0] = theTimeiohmm.bindingData.regNames[nrow]
							+ " "
							+ theTimeiohmm.bindingData.existingBindingValuesSorted[nel];
					int ntotal = theTimeiohmm.filteredClassifier.nBaseCount[nrow][theTimeiohmm.bindingData.signedBindingValuesSorted[nel]
							+ theTimeiohmm.filteredClassifier.noffset];
					tabledata[nrowindex][1] = "" + ntotal;

					for (int ngeneindex = 0; ngeneindex < bindingpvalTFindex[nrow].length; ngeneindex++) {
						int ngene = bindingpvalTFindex[nrow][ngeneindex];
						if ((include[ngene])
								&& (bindingpvalTF[nrow][ngeneindex] == theTimeiohmm.bindingData.existingBindingValuesSorted[nel])) {
							ncount++;
						}
					}

					tabledata[nrowindex][2] = "" + ncount;
					double dexpect;

					dexpect = ntotal * numgenes
							/ (double) theTimeiohmm.ntotalcombined;

					tabledata[nrowindex][3] = nf2.format(dexpect);
					double ddiff = ncount - dexpect;
					tabledata[nrowindex][4] = nf2.format(ddiff);
					tabledata[nrowindex][5] = ""
							+ DREMGui_EdgeTable.doubleToSz(StatUtil
									.hypergeometrictail(ncount - 1, ntotal,
											theTimeiohmm.ntotalcombined
													- ntotal, numgenes));

					nrowindex++;
				}
			}
		}

		sorter = new TableSorter(new TableModelST(tabledata, columnNames));
		final JTable table = new JTable(sorter);

		sorter.setTableHeader(table.getTableHeader());
		table.setPreferredScrollableViewportSize(new Dimension(800, Math.min(
				(table.getRowHeight() + table.getRowMargin())
						* table.getRowCount(), 400)));

		table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		TableColumn column;
		column = table.getColumnModel().getColumn(0);
		column.setPreferredWidth(150);
		column = table.getColumnModel().getColumn(1);
		column.setPreferredWidth(100);

		// Create the scroll pane and add the table to it.
		JScrollPane scrollPane = new JScrollPane(table);

		// Add the scroll pane to this panel.
		add(scrollPane);

		copyButton = new JButton("Copy Table", Util
				.createImageIcon("Copy16.gif"));
		copyButton.setActionCommand("copy");
		copyButton.setMinimumSize(new Dimension(800, 20));
		copyButton.addActionListener(this);

		saveButton = new JButton("Save Table", Util
				.createImageIcon("Save16.gif"));
		saveButton.setActionCommand("save");
		saveButton.setMinimumSize(new Dimension(800, 20));
		saveButton.addActionListener(this);

		JPanel buttonPanel = new JPanel();
		buttonPanel.setBackground(Color.white);
		buttonPanel.add(copyButton);
		buttonPanel.add(saveButton);

		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel.add(helpButton);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);
	}

	/**
	 * Responds to actions on the interface
	 */
	public void actionPerformed(ActionEvent e) {
		String szCommand = e.getActionCommand();

		if (szCommand.equals("copy")) {
			writeToClipboard();
		} else if (szCommand.equals("save")) {
			try {
				int nreturnVal = DREM_IO.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = DREM_IO.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					printFile(pw);
					pw.close();
				}
			} catch (FileNotFoundException ex) {
				final FileNotFoundException fex = ex;
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(null, fex.getMessage(),
								"Exception thrown", JOptionPane.ERROR_MESSAGE);
					}
				});
				ex.printStackTrace(System.out);
			}
		} else if (szCommand.equals("help")) {

			String szMessage = "A TF-summary table provides aggregate TF-gene interaction information "
					+ "for the Gene Table. The table has six columns. The columns are as follows:\n\n"
					+ "* TF - The name of the transcription factor and the value of the annotation for the TF. Only non-zero ('1' "
					+ "or '-1') annotations are included.\n\n"
					+ "* Total Overall - The number of interactions for the transcription factor of the specified value in the TF "
					+ "column among genes in the file.\n\n"
					+ "* Selected - The number of interactions of the transcription factor of the specified value in the TF column "
					+ "among genes that were in the Gene Table.\n\n"
					+ "* Expected Overall - The expected number of interactions of that value for a random set of genes the same "
					+ "size as in the Gene Table. This is the number of genes in the table times the value in Total Overall divided "
					+ "by the total number of genes in the expression data.\n\n"
					+ "* Diff Overall - The difference between Selected and Expected Overall.\n\n"
					+ "* Overall Score - The hypergeometric distribution probability of seeing a greater value than Selected.  Note "
					+ "if the TF data was used to learn the model it does not represent a true p-value, but lower values still mean "
					+ "a more significant association.\n.";

			Util.renderDialog(theFrame, szMessage);
		}
	}

}
