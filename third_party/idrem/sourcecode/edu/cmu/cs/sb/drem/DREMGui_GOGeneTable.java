package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;

import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JTable;
import java.awt.Dimension;
import java.util.*;
import java.text.NumberFormat;
import java.awt.*;
import java.awt.event.*;
import java.io.*;
import javax.swing.*;
import javax.swing.table.*;
import java.awt.datatransfer.*;

/**
 * Class encapsulates a table of genes all assigned to the same GO category when
 * clicking on a GO category in a GOTable
 */
public class DREMGui_GOGeneTable extends JPanel implements ActionListener {
	DREM_DataSet theDataSet;
	String szGoID;

	boolean[] include;
	boolean[] goinclude;
	DREM_Timeiohmm theTimeiohmm;
	Vector<String> columnNames;
	Vector<Vector<String>> tabledata;
	JButton staticButton;
	JButton saveButton, savenamesButton;
	JButton copyButton, copynamesButton;
	JButton selectButton, unselectButton;
	TableSorter sorter;
	String szTitle;
	final static Color bg = Color.white;
	final static Color fg = Color.black;
	JFrame theFrame;
	DREMGui theDREMGui;
	int numgoinclude;

	String[] tfNames;
	double[][] bindingpval;
	int[][] bindingpvalIndex;
	double[][] bindingpvalTF;
	int[][] bindingpvalTFIndex;

	/**
	 * Class constructor - builds the table
	 */
	public DREMGui_GOGeneTable(DREMGui theDREMGui, JFrame theFrame,
			DREM_DataSet theDataSet, DREM_Timeiohmm theTimeiohmm,
			double[][] bindingpval, int[][] bindingpvalIndex,
			double[][] bindingpvalTF, int[][] bindingpvalTFIndex,
			String[] tfNames, String szGoID, boolean[] include) {
		this.theFrame = theFrame;
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bg);
		setForeground(fg);
		this.theDataSet = theDataSet;
		this.szGoID = szGoID;
		this.bindingpval = bindingpval;
		this.bindingpvalIndex = bindingpvalIndex;
		this.bindingpvalTF = bindingpvalTF;
		this.bindingpvalTFIndex = bindingpvalTFIndex;
		this.tfNames = tfNames;
		this.theDREMGui = theDREMGui;
		this.include = include;
		numgoinclude = 0;
		goinclude = new boolean[include.length];
		this.theTimeiohmm = theTimeiohmm;
		columnNames = new Vector<String>();

		columnNames.add(theDataSet.szGeneHeader);
		columnNames.add(theDataSet.szProbeHeader);

		for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++) {
			columnNames.add("" + theDataSet.dsamplemins[ncolindex]);
		}

		int nstart = theDataSet.numcols + 2;
		for (int ncolindex = 0; ncolindex < tfNames.length; ncolindex++) {
			columnNames.add(tfNames[ncolindex]);
		}

		tabledata = new Vector<Vector<String>>();

		loadTable();

		sorter = new TableSorter(new TableModelST(tabledata, columnNames));
		final JTable table = new JTable(sorter);
		sorter.setTableHeader(table.getTableHeader());
		table.setPreferredScrollableViewportSize(new Dimension(800, Math.min(
				(table.getRowHeight() + table.getRowMargin())
						* table.getRowCount(), 300)));

		table.setAutoResizeMode(JTable.AUTO_RESIZE_OFF);
		TableColumn column;
		column = table.getColumnModel().getColumn(0);
		column.setPreferredWidth(150);

		column = table.getColumnModel().getColumn(1);
		column.setPreferredWidth(100);

		for (int ncolindex = 0; ncolindex < theDataSet.numcols; ncolindex++) {
			column = table.getColumnModel().getColumn(ncolindex + 2);
			column.setPreferredWidth(45);
		}

		for (int ncolindex = 0; ncolindex < tfNames.length; ncolindex++) {
			column = table.getColumnModel().getColumn(ncolindex + nstart);
			column.setPreferredWidth(45);
		}

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

		add(scrollPane);

		copynamesButton = new JButton("Copy Gene Names", Util
				.createImageIcon("Copy16.gif"));
		copynamesButton.setActionCommand("copynames");
		copynamesButton.setMinimumSize(new Dimension(800, 20));
		copynamesButton.addActionListener(this);

		savenamesButton = new JButton("Save Gene Names", Util
				.createImageIcon("Save16.gif"));
		savenamesButton.setActionCommand("savenames");
		savenamesButton.setMinimumSize(new Dimension(800, 20));
		savenamesButton.addActionListener(this);

		JPanel buttonPanel = new JPanel();
		buttonPanel.add(copyButton);
		buttonPanel.add(saveButton);
		buttonPanel.setBackground(Color.white);
		buttonPanel.add(copynamesButton);
		buttonPanel.add(savenamesButton);

		staticButton = new JButton("TF Summary");
		staticButton.setActionCommand("staticsummary");
		staticButton.setMinimumSize(new Dimension(800, 20));
		staticButton.addActionListener(this);
		buttonPanel.add(staticButton);

		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel.add(helpButton);

		selectButton = new JButton("Select by this GO category");
		selectButton.setActionCommand("select");
		selectButton.setMinimumSize(new Dimension(800, 20));
		selectButton.addActionListener(this);

		unselectButton = new JButton("Unapply GO Selection Constraints");
		unselectButton.setActionCommand("unselect");
		unselectButton.setMinimumSize(new Dimension(800, 20));
		unselectButton.setEnabled(false);
		unselectButton.addActionListener(this);

		JPanel selectGOPanel = new JPanel();
		selectGOPanel.add(selectButton);
		selectGOPanel.setBackground(Color.white);
		selectGOPanel.add(unselectButton);

		add(buttonPanel);
		add(selectGOPanel);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		selectGOPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
	}

	/**
	 * Writes the content of the table to a file
	 */
	public void printFile(PrintWriter pw) {
		int ncnsminus1 = columnNames.size() - 1;
		for (int ncol = 0; ncol < ncnsminus1; ncol++) {
			pw.print(columnNames.get(ncol) + "\t");
		}
		pw.println(columnNames.get(ncnsminus1));

		for (int nrow = 0; nrow < tabledata.size(); nrow++) {
			for (int ncol = 0; ncol < ncnsminus1; ncol++) {
				pw.print(sorter.getValueAt(nrow, ncol) + "\t");
			}
			pw.println(sorter.getValueAt(nrow, ncnsminus1));
		}
	}

	/**
	 * Copies the content of the table to the clipboard
	 */
	public void writeToClipboard() {
		StringBuffer sbuf = new StringBuffer();
		int ncnsminus1 = columnNames.size() - 1;
		for (int ncol = 0; ncol < ncnsminus1; ncol++) {
			sbuf.append(columnNames.get(ncol) + "\t");
		}
		sbuf.append(columnNames.get(ncnsminus1) + "\n");

		for (int nrow = 0; nrow < tabledata.size(); nrow++) {
			for (int ncol = 0; ncol < ncnsminus1; ncol++) {
				sbuf.append(sorter.getValueAt(nrow, ncol) + "\t");
			}
			sbuf.append(sorter.getValueAt(nrow, ncnsminus1) + "\n");
		}

		// get the system clipboard
		Clipboard systemClipboard = Toolkit.getDefaultToolkit()
				.getSystemClipboard();
		// set the textual content on the clipboard to our
		// Transferable object
		Transferable transferableText = new StringSelection(sbuf.toString());
		systemClipboard.setContents(transferableText, null);
	}

	/**
	 * Writes the names of the genes in the table to a file
	 */
	public void printGeneNames(PrintWriter pw) {
		int nsize = tabledata.size();

		for (int nrow = 0; nrow < nsize; nrow++) {
			pw.println(sorter.getValueAt(nrow, 0));
		}
	}

	/**
	 * Copies the names of the genes in the table to a clipboard
	 */
	public void writenamesToClipboard() {
		StringBuffer sbuf = new StringBuffer();
		int nsize = tabledata.size();

		for (int nrow = 0; nrow < nsize; nrow++) {
			sbuf.append(sorter.getValueAt(nrow, 0) + "\n");
		}

		// get the system clipboard
		Clipboard systemClipboard = Toolkit.getDefaultToolkit()
				.getSystemClipboard();
		Transferable transferableText = new StringSelection(sbuf.toString());
		systemClipboard.setContents(transferableText, null);
	}

	/**
	 * Helper method to put the genes into the table
	 */
	private void loadTable() {
		NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);

		for (int ngeneindex = 0; ngeneindex < include.length; ngeneindex++) {
			if (include[ngeneindex]) {
				Vector<String> rec = new Vector<String>(2);

				HashSet<String> goList;

				goList = theDataSet.tga
						.labelsForID(theDataSet.genenames[ngeneindex]);

				if ((goList.size() >= 1) && (goList.contains(szGoID))) {
					goinclude[ngeneindex] = true;
					numgoinclude++;
					rec.add(theDataSet.genenames[ngeneindex]);
					rec.add(theDataSet.probenames[ngeneindex]);
					for (int ncol = 0; ncol < theDataSet.numcols; ncol++) {
						if (theDataSet.pmavalues[ngeneindex][ncol] == 0) {
							rec.add("");
						} else {
							rec.add(nf2
									.format(theDataSet.data[ngeneindex][ncol]));
						}
					}

					int nhitindex = 0;
					for (int ncol = 0; ncol < tfNames.length; ncol++) {
						// TODO: Could this be done better by a binary search???
						while ((nhitindex < bindingpvalIndex[ngeneindex].length)
								&& (bindingpvalIndex[ngeneindex][nhitindex] < ncol)) {
							nhitindex++;
						}
						if ((nhitindex < bindingpvalIndex[ngeneindex].length)
								&& (ncol == bindingpvalIndex[ngeneindex][nhitindex])) {
							rec.add(""
									+ (int) bindingpval[ngeneindex][nhitindex]);
						} else {
							rec.add("0");
						}
					}
					tabledata.add(rec);
				}
			}
		}
	}

	/**
	 * Responds to buttons being pressed on the table
	 */
	public void actionPerformed(ActionEvent e) {
		String szCommand = e.getActionCommand();

		if (szCommand.equals("copy")) {
			writeToClipboard();
		} else if (szCommand.equals("copynames")) {
			writenamesToClipboard();
		} else if (szCommand.equals("staticsummary")) {
			javax.swing.SwingUtilities.invokeLater(new Runnable() {
				public void run() {
					String szTitle = "Static Summary";
					JFrame frame = new JFrame(szTitle);
					frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
					frame.setLocation(40, 80);
					DREMGui_StaticSummaryTable newContentPane = new DREMGui_StaticSummaryTable(
							theFrame, theDataSet, theTimeiohmm, bindingpvalTF,
							bindingpvalTFIndex, tfNames, goinclude,
							numgoinclude);

					newContentPane.setOpaque(true); // content panes must be
					// opaque
					frame.setContentPane(newContentPane);
					// Display the window.
					frame.pack();
					frame.setVisible(true);
				}
			});
		} else if ((szCommand.equals("save"))
				|| (szCommand.equals("savenames"))) {
			try {
				int nreturnVal = DREM_IO.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = DREM_IO.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("save")) {
						printFile(pw);
					} else {
						printGeneNames(pw);
					}
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
		} else if (szCommand.equals("select")) {
			theDREMGui.selectGO(szGoID);
			unselectButton.setEnabled(true);

			if (theDREMGui.filterGOFrame != null) {
				((DREMGui_GOFilter) theDREMGui.filterGOFrame.getContentPane()).rowSM
						.clearSelection();
			}
		} else if (szCommand.equals("unselect")) {
			unselectButton.setEnabled(false);
			theDREMGui.unselectGO();
		} else if (szCommand.equals("help")) {
			String szMessage = "The table contains genes assigned to the selected genes and GO Category ";
			String szlogword = "";
			if (theDataSet.btakelog) {
				szlogword = "log base two ";
			}

			szMessage += ".\n\n The first two columns of the table are: \n";
			szMessage += "*  " + theDataSet.szGeneHeader
					+ " - The name of the gene.\n" + "*  "
					+ theDataSet.szProbeHeader
					+ " - The spot ID(s) associated with the gene.\n\n";

			if (columnNames.size() == 3) {
				szMessage += "The remaining column contains the "
						+ szlogword
						+ "expression change relative to the first time point.\n\n";
			}

			szMessage += "Note:\n"
					+ "+The table can be sorted by any of the columns by clicking on the column's header.\n"
					+ "+Using the 'Save Table' button the entire table can be saved, or just the gene names "
					+ "with the 'Save Gene Names' button.";

			Util.renderDialog(theFrame, szMessage);
		}
	}
}
