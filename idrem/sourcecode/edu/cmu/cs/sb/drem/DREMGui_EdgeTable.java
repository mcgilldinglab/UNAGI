package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import edu.umd.cs.piccolo.nodes.PPath;
import edu.umd.cs.piccolo.PNode;
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
import java.awt.datatransfer.*;
import javax.swing.*;
import javax.swing.table.*;
import java.math.*;

/**
 * Class for a table that shows enrichment of TF targets along a path
 */
public class DREMGui_EdgeTable extends JPanel implements ActionListener {

	DREMGui theDREMGui;
	DREM_Timeiohmm theTimeiohmm;
	JFrame theFrame;
	String[] columnNames;
	String[][] tabledata;
	JButton copyButton;
	JButton saveButton;
	JButton colorButton;
	// JButton goButton;
	TableSorter sorter;
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	DREMGui_FilterStaticModel hmst;
	int numrows;
	DREM_Timeiohmm.Treenode ptr;
	// int npathID;
	BigInteger npathID;
	int nchild;
	int ndepth;
	NumberFormat nf;
	NumberFormat nf2;
	boolean bsplit;
	boolean broot;

	/**
	 * Constructor - builds the table
	 */
	public DREMGui_EdgeTable(DREMGui theDREMGui, JFrame theFrame,
			DREM_Timeiohmm theTimeiohmm, DREM_Timeiohmm.Treenode ptr,
			BigInteger npathID, int ndepth, int nchild, int ntype, boolean broot) {
		// assuming that if called with root node then only one child so stats
		// for root will be same as first child
		this.theDREMGui = theDREMGui;
		this.theFrame = theFrame;
		this.theTimeiohmm = theTimeiohmm;
		this.ptr = ptr;
		this.npathID = npathID;
		this.ndepth = ndepth;
		this.nchild = nchild;
		this.broot = broot;

		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);

		int numtf = theTimeiohmm.bindingData.regNames.length;
		int nsize = theTimeiohmm.bindingData.signedBindingValuesUnsorted.size();
		if (theTimeiohmm.bindingData.signedBindingValuesUnsorted
				.contains(new Integer(0))) {
			nsize--;
		}
		numrows = numtf * nsize;

		bsplit = ptr.numchildren >= 2;
		int nsplitoffset;
		if (bsplit) {
			columnNames = new String[11];
			nsplitoffset = 1;
			columnNames[2] = "Num Parent";
			columnNames[7] = "Expect Split";
			columnNames[8] = "Diff. Split";
			columnNames[9] = "Score Split";
			columnNames[10] = "% Split";
		} else {
			nsplitoffset = 0;
			columnNames = new String[6];
		}

		tabledata = new String[numrows][columnNames.length];
		columnNames[0] = "TF";
		columnNames[1] = "Num Total";
		columnNames[2 + nsplitoffset] = "Num Path";
		columnNames[3 + nsplitoffset] = "Expect Overall";
		columnNames[4 + nsplitoffset] = "Diff. Overall";
		columnNames[5 + nsplitoffset] = "Score Overall";

		nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);
		int nrowindex = 0;
		for (int nrow = 0; nrow < numtf; nrow++) {
			for (int nel = 0; nel < theTimeiohmm.bindingData.signedBindingValuesSorted.length; nel++) {
				if (theTimeiohmm.bindingData.signedBindingValuesSorted[nel] != 0) {
					tabledata[nrowindex][0] = theTimeiohmm.bindingData.regNames[nrow]
							+ " "
							+ theTimeiohmm.bindingData.signedBindingValuesSorted[nel];
					tabledata[nrowindex][1] = ""
							+ theTimeiohmm.filteredClassifier.nBaseCount[nrow][theTimeiohmm.bindingData.signedBindingValuesSorted[nel]
									+ theTimeiohmm.filteredClassifier.noffset];

					tabledata[nrowindex][2 + nsplitoffset] = ""
							+ ptr.ncountvals[nrow][nchild][nel];
					tabledata[nrowindex][3 + nsplitoffset] = nf2
							.format(ptr.dexpectEdgeFull[nchild][nrowindex]);
					tabledata[nrowindex][4 + nsplitoffset] = nf2
							.format(ptr.ddiffEdgeFull[nchild][nrowindex]);
					tabledata[nrowindex][5 + nsplitoffset] = DREMGui_EdgeTable
							.doubleToSz(ptr.dpvalEdgeFull[nchild][nrowindex]);

					if (bsplit) {
						tabledata[nrowindex][2] = ""
								+ ptr.ncountTotals[nrow][nel];
						tabledata[nrowindex][7] = nf2
								.format(ptr.dexpectEdgeSplit[nchild][nrowindex]);
						tabledata[nrowindex][8] = nf2
								.format(ptr.ddiffEdgeSplit[nchild][nrowindex]);
						tabledata[nrowindex][9] = DREMGui_EdgeTable
								.doubleToSz(ptr.dpvalEdgeSplit[nchild][nrowindex]);
						if (ptr.ncountTotals[nrow][nel] == 0) {
							tabledata[nrowindex][10] = "0.00";
						} else {
							tabledata[nrowindex][10] = ""
									+ nf2
											.format(100
													* (double) ptr.ncountvals[nrow][nchild][nel]
													/ ptr.ncountTotals[nrow][nel]);
						}
					}
					nrowindex++;
				}
			}
		}

		sorter = new TableSorter(new TableModelST(tabledata, columnNames));
		if ((columnNames.length == 6) || (ntype == 2)) {
			sorter.setSortingStatus(5 + nsplitoffset, TableSorter.ASCENDING);
		} else {
			sorter.setSortingStatus(9, TableSorter.ASCENDING);
		}

		final JTable table = new JTable(sorter);

		TableColumn column;
		for (int nindex = 0; nindex < columnNames.length; nindex++) {
			column = table.getColumnModel().getColumn(nindex);
			column.setPreferredWidth(100);
		}

		sorter.setTableHeader(table.getTableHeader());

		JScrollPane scrollPane = new JScrollPane(table);
		table.setPreferredScrollableViewportSize(new Dimension(750, Math.min(
				(table.getRowHeight() + table.getRowMargin())
						* table.getRowCount(), 300)));
		add(scrollPane);

		addBottom();
	}

	/**
	 * Helper function that adds information displayed at the bottom of the
	 * table information window
	 */
	private void addBottom() {
		JPanel countPanel = new JPanel();
		JPanel labelPanel = new JPanel();
		String szcountLabel = "Total number of genes most likely going through this path is "
				+ ptr.nextptr[nchild].numPath;
		if (bsplit) {
			szcountLabel += " ("
					+ nf2.format(100 * (double) ptr.nextptr[nchild].numPath
							/ ptr.numPath) + "% of split genes)";
		}
		JLabel countLabel = new JLabel(szcountLabel);
		countPanel.setBackground(Color.white);
		countPanel.add(countLabel);
		countPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(countPanel);
		nf = NumberFormat.getInstance(Locale.ENGLISH);
		nf.setMinimumFractionDigits(3);
		nf.setMaximumFractionDigits(3);
		String szInfo;
		if (broot) {
			szInfo = "Path output distribution at "
					+ theTimeiohmm.theDataSet.dsamplemins[0]
					+ " is Normal(mu =" + nf.format(ptr.dmean) + ",sigma = "
					+ nf.format(ptr.dsigma) + ")";
		} else {
			szInfo = "Path output distribution at "
					+ theTimeiohmm.theDataSet.dsamplemins[ptr.ndepth + 1]
					+ " is Normal(mu =" + nf.format(ptr.nextptr[nchild].dmean)
					+ ",sigma = " + nf.format(ptr.nextptr[nchild].dsigma) + ")";
		}

		JLabel infoLabel = new JLabel(szInfo);
		labelPanel.setBackground(Color.white);
		labelPanel.add(infoLabel);
		labelPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(labelPanel);

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

		colorButton = new JButton("Change Color");
		colorButton.setActionCommand("color");
		colorButton.setMinimumSize(new Dimension(800, 20));
		colorButton.addActionListener(this);

		Integer colorID = (Integer) theDREMGui.htLineIDtoColorID.get(ndepth
				+ ";" + npathID);
		Color initcolor = (Color) theDREMGui.htColorIDtoColor.get(colorID);
		colorButton.setForeground(initcolor);

		buttonPanel.add(colorButton);
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
	 * Writes the content of the table to a file specified through pw
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
		// get the system clipboard
		Clipboard systemClipboard = Toolkit.getDefaultToolkit()
				.getSystemClipboard();
		// set the textual content on the clipboard to our
		// Transferable object

		Transferable transferableText = new StringSelection(sbuf.toString());
		systemClipboard.setContents(transferableText, null);
	}

	/**
	 * Responds to buttons being pressed on the interface
	 */
	public void actionPerformed(ActionEvent e) {
		String szCommand = e.getActionCommand();
		if (szCommand.equals("copy")) {
			writeToClipboard();
		} else if (szCommand.equals("color")) {

			Integer colorID = (Integer) theDREMGui.htLineIDtoColorID.get(ndepth
					+ ";" + npathID);
			Color initcolor = (Color) theDREMGui.htColorIDtoColor.get(colorID);
			Color newColor = JColorChooser.showDialog(this, "Choose Color",
					initcolor);
			if (newColor != null) {
				colorButton.setForeground(newColor);
				ArrayList linesToUpdate = (ArrayList) theDREMGui.htColorIDtoLinesList
						.get(colorID);
				int nsize = linesToUpdate.size();
				theDREMGui.htColorIDtoColor.put(colorID, newColor);
				for (int nindex = 0; nindex < nsize; nindex++) {
					PPath currPath = (PPath) linesToUpdate.get(nindex);
					if ((theDREMGui.theSelectedRec.selectedNode != currPath)
							|| (theDREMGui.theSelectedRec.bcircle)) {
						currPath.setStrokePaint(newColor);
					} else {
						theDREMGui.htColors.put(
								theDREMGui.theSelectedRec.selectedNode,
								newColor);
					}
				}

				ArrayList circlesToUpdate = (ArrayList) theDREMGui.htColorIDtoCircleList
						.get(colorID);
				if (circlesToUpdate != null) {
					nsize = circlesToUpdate.size();
					for (int nindex = 0; nindex < nsize; nindex++) {
						PNode currCircle = (PNode) circlesToUpdate.get(nindex);
						if ((theDREMGui.theSelectedRec.selectedNode != currCircle)
								|| (!theDREMGui.theSelectedRec.bcircle)) {
							currCircle.setPaint(newColor);
						} else {
							theDREMGui.htColors.put(
									theDREMGui.theSelectedRec.selectedNode,
									newColor);
						}
					}
				}
				theDREMGui.setGeneColors();
			}
		} else if (szCommand.equals("save")) {
			try {
				int nreturnVal = DREM_IO.theChooser.showSaveDialog(this);
				if (nreturnVal == JFileChooser.APPROVE_OPTION) {
					File f = DREM_IO.theChooser.getSelectedFile();
					PrintWriter pw = new PrintWriter(new FileOutputStream(f));
					if (szCommand.equals("save")) {
						printFile(pw);
					}
					pw.close();
				}
			} catch (final FileNotFoundException fex) {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						JOptionPane.showMessageDialog(null, fex.getMessage(),
								"Exception thrown", JOptionPane.ERROR_MESSAGE);
					}
				});
				fex.printStackTrace(System.out);
			}
		} else if (szCommand.equals("help")) {
			String szMessage = "This table gives information about the TFs regulating genes "
					+ "on the selected path.  Consult section 4.13 of the user manual for more details on this table.  ";

			Util.renderDialog(theFrame, szMessage);// textArea);
		}
	}

	/**
	 * Converts the value of dval to a String that is displayed on the table
	 */
	public static String doubleToSz(double dval) {
		String szexp;
		double dtempval = dval;
		int nexp = 0;

		NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(3);
		nf2.setMaximumFractionDigits(3);

		NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);
		nf1.setMinimumFractionDigits(2);
		nf1.setMaximumFractionDigits(2);

		if (dval <= 0) {
			szexp = "0.000";
		} else {
			while ((dtempval < 0.9995) && (dtempval > 0)) {
				nexp--;
				dtempval = dtempval * 10;
			}

			if (nexp < -2) {
				dtempval = Math.pow(10, Math.log(dval) / Math.log(10) - nexp);
				szexp = nf1.format(dtempval) + "e" + nexp;

			} else {
				szexp = nf2.format(dval);
			}
		}

		return szexp;
	}

}