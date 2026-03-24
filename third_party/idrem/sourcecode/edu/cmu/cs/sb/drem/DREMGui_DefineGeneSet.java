package edu.cmu.cs.sb.drem;

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
 * Class for the window used to directly specify a subset of genes to select
 */
public class DREMGui_DefineGeneSet extends JPanel implements ItemListener,
		ActionListener, ChangeListener {

	final static Color bg = Color.white;
	final static Color fg = Color.black;
	JButton colorButton;
	final static String SZDELIM = ";|,";
	GoAnnotations tga;
	TableModelSetST tms;
	boolean bcluster;
	JFrame theFrame;
	JButton unapplyButton;
	JButton hideButton;
	final static int NMAXBAD = 30;
	TableSorter sorter;
	JButton selectgenesButton;
	DREMGui theDREMGui;
	HashSet hsTFGeneList = new HashSet();
	HashSet hsList;
	boolean bsplitgolabels = false;
	JSlider theSlider;
	Hashtable theDictionary;
	JLabel pvalLabel;
	int ninitval = 30;
	JRadioButton pathButton, splitButton;
	ButtonGroup enrichmentGroup = new ButtonGroup();

	double dsetpval;

	boolean bapplyset = false;
	DREM_Timeiohmm.Treenode rootptr;

	/**
	 * Class constructor - builds the window
	 */
	public DREMGui_DefineGeneSet(JFrame theFrame, GoAnnotations tga,
			DREMGui theDREMGui, DREM_Timeiohmm.Treenode rootptr) {
		this.theFrame = theFrame;
		this.tga = tga;
		this.theDREMGui = theDREMGui;
		this.rootptr = rootptr;

		loadTFGeneList();
		makewindow();
	}

	/**
	 * Adjust the color of p-value labels for these enrichments
	 */
	void setGeneSetLabelTextColor(DREM_Timeiohmm.Treenode treeptr,
			Color newColor) {
		if (treeptr != null) {
			treeptr.genesetText.setTextPaint(newColor);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setGeneSetLabelTextColor(treeptr.nextptr[nchild], newColor);
			}
		}
	}

	/**
	 * Adds label for enrichment of the selected set of genes for paths or
	 * splits based on the specified criteria
	 */
	public void addSetLabels(DREM_Timeiohmm.Treenode treeptr) {
		if ((treeptr != null) && (bapplyset)) {
			int nmatch = 0;
			int ncandidate = 0;
			double dpval;
			HashSet hsGO;

			if (!bsplitgolabels) {
				for (int ngene = 0; ngene < treeptr.bInNode.length; ngene++) {
					if (theDREMGui.bSetVisible[ngene]) {
						if (treeptr.bInNode[ngene]) {
							nmatch++;
						}
					}
				}

				int ncategoryall = treeptr.numPath;
				int nval = nmatch - 1;

				dpval = StatUtil.hypergeometrictail(nval, ncategoryall,
						tga.numtotalgenes - ncategoryall, tga.nGeneSet);

				if (dpval <= dsetpval) {
					treeptr.szgenesetlabel = nmatch + ";"
							+ Util.doubleToSz(dpval);
				} else {
					treeptr.szgenesetlabel = "";
				}
			} else if ((treeptr.parent != null)
					&& (treeptr.parent.numchildren >= 2)) {
				int nparentbase = 0;
				int nparentcategory = 0;

				for (int ngene = 0; ngene < treeptr.bInNode.length; ngene++) {
					if (treeptr.parent.bInNode[ngene]) {
						nparentbase++;

						if (treeptr.bInNode[ngene]) {
							nparentcategory++;
						}

						if (theDREMGui.bSetVisible[ngene]) {

							ncandidate++;
							if (treeptr.bInNode[ngene]) {
								nmatch++;
							}
						}
					}
				}

				int nval = nmatch - 1;

				dpval = StatUtil.hypergeometrictail(nval, nparentcategory,
						nparentbase - nparentcategory, ncandidate);

				if (dpval <= dsetpval) {
					treeptr.szgenesetlabel = nmatch + ";"
							+ Util.doubleToSz(dpval);
				} else {
					treeptr.szgenesetlabel = "";
				}
			} else {
				treeptr.szgenesetlabel = "";
			}

			treeptr.genesetText.setText(treeptr.szgenesetlabel);
			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				addSetLabels(treeptr.nextptr[nchild]);
			}
		}
	}

	/**
	 * Responds to changes in whether enrichment labels should be based on
	 * splits or overall
	 */
	public void itemStateChanged(ItemEvent e) {
		if (pathButton.isSelected()) {
			bsplitgolabels = false;
		} else {
			bsplitgolabels = true;
		}
		addSetLabels(rootptr);
	}

	/**
	 * Responds to changes in the label significance threshold
	 */
	public void stateChanged(ChangeEvent e) {
		JSlider source = (JSlider) e.getSource();
		if (!source.getValueIsAdjusting()) {
			dsetpval = Math.pow(10, -source.getValue() / 10.0);
			pvalLabel.setText("X = " + source.getValue() / 10.0
					+ "; p-value threshold is "
					+ DREMGui_KeyInputs.doubleToSz(dsetpval));
			addSetLabels(rootptr);
		}
	}

	/**
	 * Loads into hsList the symbols for TFs in the input
	 */
	void loadTFGeneList() {
		// add space here
		hsList = new HashSet();
		for (int ninput = 0; ninput < theDREMGui.theTimeiohmm.bindingData.regNames.length; ninput++) {
			String sztf = theDREMGui.theTimeiohmm.bindingData.regNames[ninput]
					.toUpperCase(Locale.ENGLISH);

			hsList.add(sztf);
			StringTokenizer st = new StringTokenizer(sztf, SZDELIM);
			while (st.hasMoreTokens()) {
				String sztoken = st.nextToken();
				hsList.add(sztoken);
				StringTokenizer stu = new StringTokenizer(sztoken, "_");
				if (stu.countTokens() > 1) {
					hsList.add(stu.nextToken());
				}
				// need to handle unscore
			}
		}

	}

	/**
	 * Returns true if the gene is a TF and adds it to hsTFGeneList
	 */
	boolean handleInputTF(String szname) {
		String szgene = szname;
		if (hsList.contains(szgene)) {
			hsTFGeneList.add(szname);
			return true;
		} else {
			StringTokenizer st = new StringTokenizer(szgene, SZDELIM);
			while (st.hasMoreTokens()) {
				String sztoken = st.nextToken();
				if (hsList.contains(sztoken)) {
					hsTFGeneList.add(szname);
					return true;
				} else {
					StringTokenizer stu = new StringTokenizer(sztoken, "_");
					if ((stu.countTokens() > 1)
							&& (hsList.contains(stu.nextToken()))) {
						hsTFGeneList.add(szname);
					}
				}
			}
		}

		return false;
	}

	/**
	 * Toggles the set of selected genes
	 */
	void setComplement() {
		for (int nrow = 0; nrow < tms.data.length; nrow++) {
			tms.setValueAt(Boolean.valueOf(!((Boolean) tms.getValueAt(nrow, 2))
					.booleanValue()), nrow, 2);
		}
	}

	/**
	 * Selects all genes
	 */
	void setAll(boolean bval) {
		for (int nrow = 0; nrow < tms.data.length; nrow++) {
			tms.setValueAt(Boolean.valueOf(bval), nrow, 2);
		}
	}

	/**
	 * Selects all genes that are also transcription factors
	 */
	void setAllInputs(boolean bval) {
		for (int nrow = 0; nrow < tms.data.length; nrow++) {
			if (hsTFGeneList.contains(tms.data[nrow][0])) {
				tms.setValueAt(Boolean.valueOf(bval), nrow, 2);
			}
		}

		if (bval) {
			sorter.setSortingStatus(2, TableSorter.DESCENDING);
		}
	}

	/**
	 * Sets the visibility status of the p-value labels
	 */
	void setGeneSetLabelTextVisible(DREM_Timeiohmm.Treenode treeptr,
			boolean bvisible) {
		if (treeptr != null) {
			treeptr.genesetText.setVisible(bvisible);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setGeneSetLabelTextVisible(treeptr.nextptr[nchild], bvisible);
			}
		}
	}

	/**
	 * Responds to buttons being pressed on the interface
	 */
	public void actionPerformed(ActionEvent e) {
		String szcommand = e.getActionCommand();
		int nbadline = 0;

		if (szcommand.equals("color")) {
			Color newColor = JColorChooser.showDialog(this, "Choose Color",
					theDREMGui.genesetLabelColor);
			if (newColor != null) {
				theDREMGui.genesetLabelColor = newColor;
				colorButton.setForeground(newColor);
				setGeneSetLabelTextColor(rootptr, newColor);
			}
		} else if (szcommand.equals("hide")) {
			if (theDREMGui.bshowgenesetlabels) {
				theDREMGui.bshowgenesetlabels = false;
				hideButton.setText("Show Labels");
				setGeneSetLabelTextVisible(rootptr, false);
			} else {
				theDREMGui.bshowgenesetlabels = true;
				hideButton.setText("Hide Labels");
				setGeneSetLabelTextVisible(rootptr, true);
			}
		} else if (szcommand.equals("unapply")) {
			theDREMGui.bapplygenesetlabels = false;
			unapplyButton.setEnabled(false);
			hideButton.setEnabled(false);
			for (int ngene = 0; ngene < theDREMGui.bSetVisible.length; ngene++) {
				theDREMGui.bSetVisible[ngene] = true;
				boolean bvisible = theDREMGui.bglobalVisible
						&& theDREMGui.bPathVisible[ngene]
						&& theDREMGui.bGOVisible[ngene]
						&& theDREMGui.bTFVisible[ngene];
				theDREMGui.plArray[ngene].setVisible(bvisible);
				theDREMGui.plArray[ngene].setPickable(bvisible);
			}
			theDREMGui.bfiltergeneset = false;
			theDREMGui.setFilterText();
			if (theDREMGui.theGOFilter != null) {
				theDREMGui.theGOFilter.addGOLabels(rootptr);
			}
			setGeneSetLabelTextVisible(rootptr, false);
			bapplyset = false;
		} else if (szcommand.equals("selectcomplement")) {
			setComplement();
		} else if (szcommand.equals("selectall")) {
			setAll(true);
		} else if (szcommand.equals("selectnone")) {
			setAll(false);
		} else if (szcommand.equals("selecttf")) {
			setAllInputs(true);
		} else if (szcommand.equals("unselecttf")) {
			setAllInputs(false);
		} else if (szcommand.equals("save")) {
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
		} else if ((szcommand.equals("load")) || (szcommand.equals("query"))) {
			theDREMGui.bapplygenesetlabels = true;

			int returnVal = -100;
			if (szcommand.equals("load")) {
				returnVal = DREM_IO.fc.showOpenDialog(this);
				String szMissing = "";
				StringBuffer szMissingBuf = new StringBuffer();

				if (returnVal == JFileChooser.APPROVE_OPTION) {
					setAll(false);

					BufferedReader br = null;
					try {
						File file = DREM_IO.fc.getSelectedFile();
						FileReader fr = new FileReader(file);
						br = new BufferedReader(fr);
						String szLine;
						HashMap htLoadGenes = new HashMap();
						HashMap htMissing = new HashMap();

						while ((szLine = br.readLine()) != null) {
							String szname = szLine.trim().toUpperCase(
									Locale.ENGLISH);

							if (tga.htGeneNames.get(szname) != null) {
								htLoadGenes.put(szname, Boolean.valueOf(true));
							} else if ((!szname.equals("0"))
									&& (htMissing.get(szname) == null)
									&& (nbadline < NMAXBAD)) {
								szMissingBuf.append(szLine + "\n");
								htMissing.put(szname, Boolean.valueOf(true));
								nbadline++;
							}
						}

						szMissing = szMissingBuf.toString();
						for (int nrow = 0; nrow < tms.data.length; nrow++) {
							if (htLoadGenes.get(tms.data[nrow][0]) != null) {
								tms.setValueAt(Boolean.valueOf(true), nrow, 2);
							}
						}
					} catch (FileNotFoundException ex) {
						JOptionPane.showMessageDialog(this, ex.getMessage(),
								"Exception thrown", JOptionPane.ERROR_MESSAGE);
						ex.printStackTrace(System.out);
					} catch (IOException ex) {
						JOptionPane.showMessageDialog(this, ex.getMessage(),
								"Exception thrown", JOptionPane.ERROR_MESSAGE);
						ex.printStackTrace(System.out);
					} finally {
						if (br != null) {
							try {
								br.close();
							} catch (IOException ex) {
								JOptionPane.showMessageDialog(this, ex
										.getMessage(), "Exception thrown",
										JOptionPane.ERROR_MESSAGE);
								ex.printStackTrace(System.out);
							}
						}
					}

					// want to show genes loaded
					sorter.setSortingStatus(2, TableSorter.DESCENDING);
				}

				if (!szMissing.equals("")) {
					if (nbadline < NMAXBAD) {
						szMissing = "The following genes from the loaded gene "
								+ "set are not in the data file:\n" + szMissing;
					} else {
						szMissing = "The following are the first " + NMAXBAD
								+ " genes from the loaded gene "
								+ "set that are not in the data file:\n"
								+ szMissing;
					}
					JOptionPane.showMessageDialog(this, szMissing,
							"Genes Not in the Data File",
							JOptionPane.WARNING_MESSAGE);
				}
			}

			if (((szcommand.equals("load")) && (returnVal == JFileChooser.APPROVE_OPTION))
					|| (szcommand.equals("query"))) {
				String[] genenames = theDREMGui.theTimeiohmm.theDataSet.genenames;

				HashMap hsGO = tga.htGeneNames;

				for (int ngene = 0; ngene < genenames.length; ngene++) {
					if (((Boolean) hsGO.get(genenames[ngene])).booleanValue()) {
						theDREMGui.bSetVisible[ngene] = true;
						boolean bvisible = theDREMGui.bglobalVisible
								&& theDREMGui.bGOVisible[ngene]
								&& theDREMGui.bPathVisible[ngene]
								&& theDREMGui.bTFVisible[ngene];
						theDREMGui.plArray[ngene].setVisible(bvisible);
						theDREMGui.plArray[ngene].setPickable(bvisible);
					} else {
						theDREMGui.bSetVisible[ngene] = false;
						theDREMGui.plArray[ngene].setVisible(false);
						theDREMGui.plArray[ngene].setPickable(false);
					}
				}
				unapplyButton.setEnabled(true);
				theDREMGui.bfiltergeneset = true;
				theDREMGui.setFilterText();
				if (theDREMGui.theGOFilter != null) {
					theDREMGui.theGOFilter.addGOLabels(rootptr);
				}

				bapplyset = true;
				addSetLabels(rootptr);

				setGeneSetLabelTextVisible(
						rootptr,
						theDREMGui.bshowgenesetlabels
								&& (theDREMGui.bglobalnode || !theDREMGui.battachlabels));
				hideButton.setEnabled(true);
			}
		} else if (szcommand.equals("help")) {
			String szMessage = "The above dialog allows a user to select a subset of genes based on the gene names.  "
					+ "In order to select a subset one must select the corresponding boxes of the desired genes, "
					+ "and then press the Apply Selection Constraints "
					+ "button. Pressing the button Unapply Selection Constraints removes the filter based on "
					+ "the gene set but does not clear the checkboxes. When a gene set is selected labels for "
					+ "paths enriched for the gene set at a p-value determined "
					+ "by the slider appear. P-values can either be Split enrichments which uses the genes "
					+ "going into the prior split as the base set for the enrichment caculation, or Overall Enrichments "
					+ "which uses all the genes on the microarray as a base set.\n\n"
					+ "Below are a description of the additional buttons on this window:\n"
					+ "*Select All - checks all the gene boxes\n"
					+ "*Unselect All - unchecks all the gene boxes\n"
					+ "*Select Complement - checks all currently unchecked boxes and unchecks all currently checked boxes\n"
					+ "*Select All TFs - checks all the genes which also appear in a column header of the "
					+ "TF-gene interaction file\n"
					+ "*Unselect All TFs - unchecks all the genes which also appear in a column "
					+ "header of the TF-gene interaction file\n"
					+ "*Apply Selection Constraints - requires any gene that is displayed on the main "
					+ "interface to be currently checked\n"
					+ "*Unapply Selection Constraints - removes any selection requirement from the last "
					+ "time the apply selection constraints button was pressed\n"
					+ "*Change Label Colors - pressing the button opens a dialog window to change color of gene set p-value "
					+ "significance labels. The current color of the significance labels are the same of the text of the button.\n"
					+ "*Hide Labels - hides the p-value significance labels\n"
					+ "*Load Gene Set - option to select the genes listed in a file\n"
					+ "*Save Gene Set - option to export to a file the list of genes currently checked";

			Util.renderDialog(theFrame, szMessage, -200, -100);
		}
	}

	/**
	 * Writes the content of the table to a file
	 */
	public void printFile(PrintWriter pw) {
		for (int nrow = 0; nrow < tms.data.length; nrow++) {
			if (((Boolean) tms.data[nrow][2]).booleanValue()) {
				pw.println(tms.data[nrow][0]);
			}

		}
	}

	/**
	 * Helper function that builds the interface window
	 */
	private void makewindow() {
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bg);
		setForeground(fg);

		Iterator eNames = tga.htGeneNames.keySet().iterator();

		int nsize = tga.htGeneNames.size();
		Object[][] tabledata = new Object[nsize][3];
		String[] columnNames = new String[3];

		columnNames[0] = "Gene Name";
		columnNames[1] = "TF";
		columnNames[2] = "In Gene Set";
		String[] fullgenenames = new String[nsize];

		for (int nrow = 0; nrow < nsize; nrow++) {
			fullgenenames[nrow] = (String) eNames.next();
		}
		Arrays.sort(fullgenenames);

		boolean btrue = false;
		for (int nrow = 0; nrow < nsize; nrow++) {
			tabledata[nrow][0] = fullgenenames[nrow];
			tabledata[nrow][2] = tga.htGeneNames.get(fullgenenames[nrow]);
			btrue = (btrue || ((Boolean) tabledata[nrow][2]).booleanValue());
			if (handleInputTF(fullgenenames[nrow])) {
				tabledata[nrow][1] = "Yes";
			} else {
				tabledata[nrow][1] = "";
			}
		}

		tms = new TableModelSetST(tabledata, columnNames, tga);

		sorter = new TableSorter(tms);
		if (btrue) {
			sorter.setSortingStatus(2, TableSorter.DESCENDING);
		}
		final JTable table = new JTable(sorter);
		sorter.setTableHeader(table.getTableHeader());

		TableColumn column;
		table.setPreferredScrollableViewportSize(new Dimension(500, 175));
		column = table.getColumnModel().getColumn(1);
		column.setPreferredWidth(8);
		column = table.getColumnModel().getColumn(0);
		column.setPreferredWidth(200);
		column = table.getColumnModel().getColumn(2);
		column.setPreferredWidth(50);
		// Create the scroll pane and add the table to it.
		JScrollPane scrollPane = new JScrollPane(table);

		// Add the scroll pane to this panel.
		add(scrollPane);

		JLabel theTopLabel = new JLabel(
				"  Only display enrichments with a p-value less than 10^-X where X is:");
		JPanel topPanel = new JPanel();
		topPanel.add(theTopLabel);
		topPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));
		topPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(topPanel);

		pvalLabel = new JLabel("X = " + ninitval / 10.0
				+ "; p-value threshold is "
				+ DREMGui_KeyInputs.doubleToSz(Math.pow(10, -ninitval / 10.0)));
		dsetpval = Math.pow(10, -ninitval / 10.0);
		theSlider = new JSlider(0, 120, ninitval);
		theDictionary = new Hashtable();
		for (int nindex = 0; nindex <= 12; nindex++) {
			theDictionary
					.put(new Integer(nindex * 10), new JLabel("" + nindex));
		}
		theSlider.setLabelTable(theDictionary);
		theSlider.setMajorTickSpacing(10);
		theSlider.setMinorTickSpacing(5);
		theSlider.setPaintTicks(true);
		theSlider.setPaintLabels(true);
		theSlider.addChangeListener(this);
		theSlider.setPaintTicks(true);
		add(theSlider);

		JPanel labelPanel = new JPanel();
		labelPanel.add(pvalLabel);
		labelPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(labelPanel);
		labelPanel.setBackground(Color.white);

		JPanel typePanel = new JPanel();
		JLabel theGOLabel = new JLabel("p-values should be");
		typePanel.setBackground(Color.white);
		typePanel.add(theGOLabel);
		pathButton = new JRadioButton("Overall Enrichments");
		splitButton = new JRadioButton("Split Enrichments");
		enrichmentGroup.add(pathButton);
		enrichmentGroup.add(splitButton);
		pathButton.addItemListener(this);
		splitButton.addItemListener(this);
		if (bsplitgolabels) {
			splitButton.setSelected(true);
		} else {
			pathButton.setSelected(true);
		}

		pathButton.setBackground(Color.white);
		splitButton.setBackground(Color.white);
		typePanel.add(pathButton);
		typePanel.add(splitButton);
		add(typePanel);

		typePanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));

		JPanel buttonPanel = new JPanel();
		buttonPanel.setBackground(Color.white);

		JButton selectallButton = new JButton("Select All");
		selectallButton.setActionCommand("selectall");
		selectallButton.addActionListener(this);
		buttonPanel.add(selectallButton);

		JButton selectnoneButton = new JButton("Unselect All");
		selectnoneButton.setActionCommand("selectnone");
		selectnoneButton.addActionListener(this);
		buttonPanel.add(selectnoneButton);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);

		JButton selectcomplementButton = new JButton("Select Complement");
		selectcomplementButton.setActionCommand("selectcomplement");
		selectcomplementButton.addActionListener(this);
		buttonPanel.add(selectcomplementButton);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);

		JPanel buttonPanelInput = new JPanel();
		buttonPanelInput.setBackground(Color.white);

		JButton selecttfButton = new JButton("Select All TFs");
		selecttfButton.setActionCommand("selecttf");
		selecttfButton.addActionListener(this);
		buttonPanelInput.add(selecttfButton);

		JButton unselecttfButton = new JButton("Unselect All TFs");
		unselecttfButton.setActionCommand("unselecttf");
		unselecttfButton.addActionListener(this);
		buttonPanelInput.add(unselecttfButton);

		add(buttonPanelInput);

		JPanel buttonPanel2 = new JPanel();
		buttonPanel2.setBackground(Color.white);

		JPanel buttonPanel3 = new JPanel();
		buttonPanel3.setBackground(Color.white);

		JButton queryButton = new JButton("Apply  Selection Constraints");

		queryButton.setActionCommand("query");
		queryButton.addActionListener(this);
		buttonPanel3.add(queryButton);

		unapplyButton = new JButton("Unapply  Selection Constraints");
		unapplyButton.setActionCommand("unapply");
		unapplyButton.addActionListener(this);
		unapplyButton.setEnabled(false);
		buttonPanel3.add(unapplyButton);
		buttonPanel3.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel3);

		hideButton = new JButton("Hide Labels");

		hideButton.setEnabled(false);
		hideButton.setActionCommand("hide");
		hideButton.addActionListener(this);
		buttonPanel2.add(hideButton);

		colorButton = new JButton("Change Labels Color");
		colorButton.setActionCommand("color");
		colorButton.setMinimumSize(new Dimension(800, 20));
		colorButton.addActionListener(this);
		colorButton.setForeground(theDREMGui.genesetLabelColor);
		buttonPanel2.add(colorButton);

		JButton loadButton = new JButton("Load Gene Set", Util
				.createImageIcon("Open16.gif"));
		loadButton.setActionCommand("load");
		loadButton.addActionListener(this);
		buttonPanelInput.add(loadButton);

		JButton saveButton = new JButton("Save Gene Set", Util
				.createImageIcon("Save16.gif"));
		saveButton.setActionCommand("save");
		saveButton.addActionListener(this);
		buttonPanel2.add(saveButton);

		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel2.add(helpButton);
		buttonPanel2.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel2);
	}
}
