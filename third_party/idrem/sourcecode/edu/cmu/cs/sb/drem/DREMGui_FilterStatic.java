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
 * Class encapsulates window used to select a subset of genes based on the
 * static transcription factors in the input to regulate the gene.
 */
public class DREMGui_FilterStatic extends JPanel implements ActionListener,
		ItemListener, ChangeListener {
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;

	boolean bnegate = true;
	JFrame theFrame;
	JButton colorButton;
	JRadioButton intersectButton, unionButton;
	double[] tfVals;

	DREMGui_FilterStaticModel hmst;
	TableSorter sorter;
	DREM_Timeiohmm theTimeiohmm;
	DREMGui theDREMGui;
	int numrows, numcols;
	JButton unapplyButton;
	JButton hideButton;
	boolean bsplitgolabels = true;
	JSlider theSlider;
	Hashtable theDictionary;
	JLabel pvalLabel;
	int ninitval = 30;
	JRadioButton pathButton, splitButton;
	ButtonGroup enrichmentGroup = new ButtonGroup();
	double dsetpval;
	GoAnnotations tga;
	boolean bapplyset = false;
	boolean boveralltfeligible;
	int noveralltfrow;
	int noveralltfcol;

	DREM_Timeiohmm.Treenode rootptr;

	JCheckBox complementcheck = new JCheckBox(
			"Use Complement of Above Criteria", false);

	/**
	 * Class constructor - builds the interface window
	 */
	public DREMGui_FilterStatic(JFrame theFrame, DREMGui theDREMGui,
			GoAnnotations tga, DREM_Timeiohmm.Treenode rootptr) {
		this.theFrame = theFrame;
		this.theDREMGui = theDREMGui;
		this.tga = tga;
		this.theTimeiohmm = theDREMGui.theTimeiohmm;
		this.rootptr = rootptr;
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);
		numcols = theTimeiohmm.bindingData.existingBindingValuesUnsorted.size() + 1;

		tfVals = theTimeiohmm.bindingData.existingBindingValuesSorted;
		numrows = theTimeiohmm.bindingData.regNames.length;

		Object[][] tabledata = new Object[numrows][numcols];
		String[] columnNames = new String[numcols];

		columnNames[0] = "Transcription Factor";
		for (int ncol = 1; ncol < numcols; ncol++) {
			columnNames[ncol] = "" + tfVals[ncol - 1];
		}

		for (int nrow = 0; nrow < numrows; nrow++) {
			tabledata[nrow][0] = theTimeiohmm.bindingData.regNames[nrow];
			for (int ncol = 1; ncol < numcols; ncol++) {
				tabledata[nrow][ncol] = Boolean.valueOf(false);
			}
		}

		hmst = new DREMGui_FilterStaticModel(tabledata, columnNames);
		sorter = new TableSorter(hmst);
		final JTable table = new JTable(sorter);
		sorter.setTableHeader(table.getTableHeader());

		TableColumn column;
		column = table.getColumnModel().getColumn(0);
		column.setPreferredWidth(125);
		for (int ncol = 1; ncol < numcols; ncol++) {
			column = table.getColumnModel().getColumn(ncol);
			column.setPreferredWidth(50);
		}

		// Create the scroll pane and add the table to it.
		JScrollPane scrollPane = new JScrollPane(table);
		table.setPreferredScrollableViewportSize(new Dimension(425, Math.min(
				(table.getRowHeight() + table.getRowMargin())
						* table.getRowCount(), 200)));
		// Add the scroll pane to this panel.
		add(scrollPane);

		JPanel buttonPanel0 = new JPanel();
		buttonPanel0.setBackground(Color.white);
		unionButton = new JRadioButton("at least one TF");
		unionButton.setBackground(Color.white);

		intersectButton = new JRadioButton("all TFs");
		intersectButton.setBackground(Color.white);
		unionButton.setSelected(true);
		ButtonGroup group = new ButtonGroup();
		group.add(intersectButton);
		group.add(unionButton);
		buttonPanel0
				.add(new JLabel("Genes selected must meet constraints for"));
		buttonPanel0.add(intersectButton);
		buttonPanel0.add(unionButton);
		buttonPanel0.setMaximumSize(new Dimension(Integer.MAX_VALUE, 12));
		add(buttonPanel0);
		JPanel complementPanel = new JPanel();
		complementPanel.add(complementcheck);
		complementPanel.setBackground(Color.white);
		complementcheck.setBackground(Color.white);
		complementPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 12));
		add(complementPanel);

		JLabel theTopLabel = new JLabel(
				"  Only display enrichments with a score less than 10^-X where X is:");

		JPanel topPanel = new JPanel();
		topPanel.add(theTopLabel);
		topPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));
		topPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(topPanel);

		pvalLabel = new JLabel("X = " + ninitval / 10.0
				+ "; score threshold is "
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
		JLabel theGOLabel = new JLabel("Score should be");
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

		hideButton = new JButton("Hide Labels");
		hideButton.setEnabled(false);
		hideButton.setActionCommand("hide");
		hideButton.addActionListener(this);
		buttonPanel.add(hideButton);

		colorButton = new JButton("Change Labels Color");
		colorButton.setActionCommand("color");
		colorButton.setMinimumSize(new Dimension(800, 20));
		colorButton.addActionListener(this);
		colorButton.setForeground(theDREMGui.tfLabelColor);
		buttonPanel.add(colorButton);

		JButton selectnoneButton = new JButton("Unselect All");
		selectnoneButton.setActionCommand("selectnone");
		selectnoneButton.addActionListener(this);
		buttonPanel.add(selectnoneButton);

		JButton selectallButton = new JButton("Select All");
		selectallButton.setActionCommand("selectall");
		selectallButton.addActionListener(this);
		buttonPanel.add(selectallButton);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel);
		JPanel buttonPanel2 = new JPanel();
		buttonPanel2.setBackground(Color.white);

		JButton queryButton = new JButton("Apply Selection Constraints");
		queryButton.setActionCommand("applyselect");
		queryButton.addActionListener(this);
		buttonPanel2.add(queryButton);
		buttonPanel2.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel2);

		unapplyButton = new JButton("Unapply Selection Constraints");
		unapplyButton.setActionCommand("unapply");
		unapplyButton.addActionListener(this);
		unapplyButton.setEnabled(false);
		buttonPanel2.add(unapplyButton);

		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel2.add(helpButton);
		buttonPanel2.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel2);
		buttonPanel.setBackground(Color.white);
	}

	/**
	 * Sets the selection status of all checkboxes to the value of bval
	 */
	void setAll(boolean bval) {
		for (int nrow = 0; nrow < numrows; nrow++) {
			for (int ncol = 1; ncol < numcols; ncol++) {
				hmst.setValueAt(Boolean.valueOf(bval), nrow, ncol);
			}
		}
	}

	/**
	 * Sets whether the significance labels are visible or not
	 */
	void setTFSetLabelTextVisible(DREM_Timeiohmm.Treenode treeptr,
			boolean bvisible) {
		if (treeptr != null) {
			treeptr.tfsetText.setVisible(bvisible);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setTFSetLabelTextVisible(treeptr.nextptr[nchild], bvisible);
			}
		}
	}

	/**
	 * Sets the color of the significance labels
	 */
	void setTFSetLabelTextColor(DREM_Timeiohmm.Treenode treeptr, Color newColor) {
		if (treeptr != null) {
			treeptr.tfsetText.setTextPaint(newColor);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setTFSetLabelTextColor(treeptr.nextptr[nchild], newColor);
			}
		}
	}

	/**
	 * Responds to buttons being pressed on the input interface
	 */
	public void actionPerformed(ActionEvent e) {

		String szcommand = e.getActionCommand();
		int nbadline = 0;

		if (szcommand.equals("color")) {
			Color newColor = JColorChooser.showDialog(this, "Choose Color",
					theDREMGui.tfLabelColor);
			if (newColor != null) {
				theDREMGui.tfLabelColor = newColor;
				colorButton.setForeground(newColor);
				setTFSetLabelTextColor(rootptr, newColor);
			}
		} else if (szcommand.equals("hide")) {
			if (theDREMGui.bshowtfsetlabels) {
				theDREMGui.bshowtfsetlabels = false;
				hideButton.setText("Show Labels");
				setTFSetLabelTextVisible(rootptr, false);
			} else {
				theDREMGui.bshowtfsetlabels = true;
				hideButton.setText("Hide Labels");
				setTFSetLabelTextVisible(rootptr, true);
			}
		} else if (szcommand.equals("unapply")) {
			unapplyButton.setEnabled(false);
			for (int ngene = 0; ngene < theTimeiohmm.bindingData.gene2RegMaxBinding.length; ngene++) {
				theDREMGui.bTFVisible[ngene] = true;
				boolean bvisible = theDREMGui.bglobalVisible
						&& theDREMGui.bPathVisible[ngene]
						&& theDREMGui.bGOVisible[ngene]
						&& theDREMGui.bSetVisible[ngene];
				theDREMGui.plArray[ngene].setVisible(bvisible);
				theDREMGui.plArray[ngene].setPickable(bvisible);
			}
			theDREMGui.bfilterinput = false;
			theDREMGui.setFilterText();
			if (theDREMGui.theGOFilter != null) {
				theDREMGui.theGOFilter.addGOLabels(rootptr);
			}
			setTFSetLabelTextVisible(rootptr, false);
			theDREMGui.bapplytfsetlabels = false;
			bapplyset = false;
		} else if (szcommand.equals("help")) {
			String szMessage = "This dialog box allows the selection of a subset of genes based on being regulated "
					+ "by a common transcription factor (TF) or combination of TFs. For each TF from "
					+ "the 'TF-gene Interactions File', there is a checkbox for the values of '0' "
					+ "and '1'.  If '-1' values are also present in the 'TF-gene Interactions File', "
					+ "then there are also checkboxes for this value. "
					+ "If the option 'Genes selected must meet constraints for' is "
					+ "set to 'all TFs', then only genes which have TF-gene "
					+ "interaction values matching a checked box value for all TFs will be selected. "
					+ "In this case at least one value must be specified for every TF otherwise "
					+ "it is not possible to have a match.  If the option is set to 'at least one TF', then any "
					+ "gene with a predicted TF-gene regulation interaction that matches a checked "
					+ "box for at least one TF will be selected.  If the option "
					+ "'Use Complement of Above Criteria' is selected the complement "
					+ "of the set of genes described by the above criteria will be selected.  "
					+ "To actually apply changes made "
					+ "to the checkboxes the button 'Apply Selection Constraints' must be pressed.  "
					+ "Pressing the button 'Unapply Selection Constraints' removes selection "
					+ "constraints based on TF-gene regulation interactions.  "
					+ "To have all the checkboxes selected press the button 'Select All', and to "
					+ "have no checkboxes selected press the button 'Unselect All'.\n\n"
					+ "In addition to selecting genes, when the 'Apply Selection Constraints' button is pressed "
					+ "labels appear when the score for any set of genes is less than the score "
					+ "threshold determined by the setting of the slider "
					+ "under 'Only display enrichments with a score less than 10^{-X} where X is'.  "
					+ "The score can be based on 'Split Enrichments' or 'Overall Enrichments' "
					+ "for genes regulated by the selected TF regulation constraints.  "
					+ "Split enrichments are computed based on the hypergeometric distribution where "
					+ "the base set of genes are all genes going into the prior split on the path.  "
					+ "The base set of genes for 'Overall Enrichments' "
					+ "is all genes included in the expression data file or the 'Pre-filtered Gene File'.  "
					+ "Overall enrichments are currently only supported when selecting by a single TF.  "
					+ "Labels appear to the immediate right of the first node on the path out of the split.  "
					+ "The label contains the number of genes and then the score separated by a semi-colon.  "
					+ "To hide labels press the 'Hide Labels' button.  When the labels are hidden "
					+ "the button now reads 'Show Labels', and pressing it reverts the labels to being "
					+ "shown again.  The color of labels can be changed through the 'Change Labels Color' "
					+ "button.  The color of the TF labels will match that of the color of the text of this "
					+ "'Change Labels Color' button.";

			Util.renderDialog(theFrame, szMessage, -200, -100);
		} else if (szcommand.equals("applyselect")) {
			bnegate = complementcheck.isSelected();
			theDREMGui.bapplytfsetlabels = true;
			// build hash set of selected
			unapplyButton.setEnabled(true);
			HashSet[] hsSelected = new HashSet[numrows];
			boveralltfeligible = (!bnegate);
			noveralltfrow = -1;

			for (int nrow = 0; nrow < numrows; nrow++) {
				hsSelected[nrow] = new HashSet();
				for (int ncol = 1; ncol < numcols; ncol++) {
					if (((Boolean) hmst.getValueAt(nrow, ncol)).booleanValue()) {
						hsSelected[nrow].add(new Double(tfVals[ncol - 1]));
						if (noveralltfrow == -1) {
							noveralltfrow = nrow;
							noveralltfcol = ncol - 1;
						} else {
							boveralltfeligible = false;
						}
					}
				}
				theDREMGui.bfilterinput = true;
				theDREMGui.setFilterText();
			}

			boolean bintersect = intersectButton.isSelected();
			if (bintersect) {
				boveralltfeligible = false;
				for (int ngene = 0; ngene < theTimeiohmm.bindingData.gene2RegMaxBinding.length; ngene++) {
					// just one hit needed
					boolean bsatisfied = true;
					int nrow = 0;
					while ((nrow < numrows) && (bsatisfied)) {
						// assumes array is sorted
						int nhit = Arrays
								.binarySearch(
										theTimeiohmm.bindingData.gene2RegMaxBindingIndex[ngene],
										nrow);
						double dval;
						if (nhit < 0) {
							dval = 0;
						} else {
							dval = theTimeiohmm.bindingData.gene2RegMaxBinding[ngene][nhit];
						}

						if (!hsSelected[nrow].contains(new Double(dval))) {
							bsatisfied = false;
						} else {
							nrow++;
						}
					}

					if (bnegate) {
						bsatisfied = !bsatisfied;
					}

					if (!bsatisfied) {
						theDREMGui.bTFVisible[ngene] = false;
						theDREMGui.plArray[ngene].setVisible(false);
						theDREMGui.plArray[ngene].setPickable(false);
					} else {
						theDREMGui.bTFVisible[ngene] = true;
						boolean bvisible = theDREMGui.bglobalVisible
								&& theDREMGui.bPathVisible[ngene]
								&& theDREMGui.bSetVisible[ngene]
								&& theDREMGui.bGOVisible[ngene];
						theDREMGui.plArray[ngene].setVisible(bvisible);
						theDREMGui.plArray[ngene].setPickable(bvisible);
					}
				}

			} else {
				for (int ngene = 0; ngene < theTimeiohmm.bindingData.gene2RegMaxBinding.length; ngene++) {
					// just one hit needed
					boolean bsatisfied = false;
					int nrow = 0;
					while ((nrow < numrows) && (!bsatisfied)) {
						// assumes array is sorted
						int nhit = Arrays
								.binarySearch(
										theTimeiohmm.bindingData.gene2RegMaxBindingIndex[ngene],
										nrow);
						double dval;
						if (nhit < 0) {
							dval = 0;
						} else {
							dval = theTimeiohmm.bindingData.gene2RegMaxBinding[ngene][nhit];
						}

						if (hsSelected[nrow].contains(new Double(dval))) {
							bsatisfied = true;
						} else {
							nrow++;
						}
					}

					if (bnegate) {
						bsatisfied = !bsatisfied;
					}

					if (!bsatisfied) {
						theDREMGui.bTFVisible[ngene] = false;
						theDREMGui.plArray[ngene].setVisible(false);
						theDREMGui.plArray[ngene].setPickable(false);
					} else {
						theDREMGui.bTFVisible[ngene] = true;
						boolean bvisible = theDREMGui.bglobalVisible
								&& theDREMGui.bPathVisible[ngene]
								&& theDREMGui.bSetVisible[ngene]
								&& theDREMGui.bGOVisible[ngene];
						theDREMGui.plArray[ngene].setVisible(bvisible);
						theDREMGui.plArray[ngene].setPickable(bvisible);
					}
				}
			}

			if (theDREMGui.theGOFilter != null) {
				theDREMGui.theGOFilter.addGOLabels(rootptr);
			}
			bapplyset = true;
			addSetLabels(rootptr);
			setTFSetLabelTextVisible(rootptr, theDREMGui.bshowtfsetlabels
					&& (theDREMGui.bglobalnode || !theDREMGui.battachlabels));
			hideButton.setEnabled(true);
		} else if (szcommand.equals("selectall")) {
			setAll(true);
		} else if (szcommand.equals("selectnone")) {
			setAll(false);
		}
	}

	/**
	 * Adds significance labels to the main interface based on the specificed
	 * settings of the parameters
	 */
	public void addSetLabels(DREM_Timeiohmm.Treenode treeptr) {
		if ((treeptr != null) && (bapplyset)) {
			int nmatch = 0;
			int ncandidate = 0;
			double dpval = -1;
			HashSet hsGO;

			if (!bsplitgolabels) {
				for (int ngene = 0; ngene < treeptr.bInNode.length; ngene++) {
					if (theDREMGui.bTFVisible[ngene]) {
						if (treeptr.bInNode[ngene]) {
							nmatch++;
						}
					}
				}

				int ncategoryall = treeptr.numPath;

				if (boveralltfeligible) {
					int nsumfullel = theTimeiohmm.filteredClassifier.nBaseCount[noveralltfrow][noveralltfcol];
					int nsumthispath = treeptr.numPath;

					dpval = StatUtil.hypergeometrictail(nmatch - 1, nsumfullel,
							theTimeiohmm.ntotalcombined - nsumfullel,
							nsumthispath);

					if (dpval <= dsetpval) {
						treeptr.sztfsetlabel = nmatch + ";"
								+ Util.doubleToSz(dpval);

					} else {
						treeptr.sztfsetlabel = "";
					}

				} else {
					if (dpval <= dsetpval) {
						treeptr.sztfsetlabel = nmatch + ";";
					} else {
						treeptr.sztfsetlabel = "";
					}
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

						if (theDREMGui.bTFVisible[ngene]) {

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
					treeptr.sztfsetlabel = nmatch + ";"
							+ Util.doubleToSz(dpval);
				} else {
					treeptr.sztfsetlabel = "";
				}
			} else {
				treeptr.sztfsetlabel = "";
			}

			treeptr.tfsetText.setText(treeptr.sztfsetlabel);
			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				addSetLabels(treeptr.nextptr[nchild]);
			}
		}
	}

	/**
	 * Responds to a change in the selection of the radio button
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
	 * Responds to changes in the slider specifying the significance threshold
	 */
	public void stateChanged(ChangeEvent e) {
		JSlider source = (JSlider) e.getSource();
		if (!source.getValueIsAdjusting()) {
			dsetpval = Math.pow(10, -source.getValue() / 10.0);
			pvalLabel.setText("X = " + source.getValue() / 10.0
					+ "; score threshold is "
					+ DREMGui_KeyInputs.doubleToSz(dsetpval));
			addSetLabels(rootptr);
		}
	}
}