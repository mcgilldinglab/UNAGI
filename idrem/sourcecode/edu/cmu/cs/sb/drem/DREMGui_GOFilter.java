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
 * Class encapsulates window used to select a subset of genes all belonging to
 * the same GO category
 */
public class DREMGui_GOFilter extends JPanel implements ActionListener,
		ItemListener, ChangeListener {
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	int ninitval = 0;
	boolean bsplitgolabels = false;
	boolean ballgenes = true;
	JSlider theSlider;
	Hashtable theDictionary;
	String szGO;

	double[] tfVals;
	DREMGui_FilterGOModel hmst;
	TableSorter sorter;
	DREM_Timeiohmm theTimeiohmm;
	DREMGui theDREMGui;
	int numrows, numcols;
	JButton unapplyButton;
	JButton hideButton;
	JFrame theFrame;
	ListSelectionModel rowSM;
	JFrame defineframe;
	JRadioButton pathButton, splitButton;
	JRadioButton allButton, selectButton;
	ButtonGroup enrichmentGroup = new ButtonGroup();
	ButtonGroup countGroup = new ButtonGroup();
	double dgopval = .001;
	JLabel pvalLabel;
	JButton colorButton;

	DREM_DataSet theDataSet;
	GoAnnotations tga;
	DREM_Timeiohmm.Treenode rootptr;

	/**
	 * Class constructor - builds the panel window
	 */
	public DREMGui_GOFilter(JFrame theFrame, DREMGui theDREMGui,
			DREM_GoAnnotations.RecIDdrem[] dremGOrecs,
			final DREM_Timeiohmm.Treenode rootptr) {
		this.theFrame = theFrame;
		this.theDREMGui = theDREMGui;
		this.theTimeiohmm = theDREMGui.theTimeiohmm;
		this.theDataSet = theDREMGui.theTimeiohmm.theDataSet;
		this.tga = this.theDataSet.tga;
		this.rootptr = rootptr;

		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);
		int ninitval = (int) Math.round(-10 * Math.log(dgopval) / Math.log(10));
		pvalLabel = new JLabel("X = " + ninitval / 10.0
				+ "; p-value threshold is "
				+ DREMGui_KeyInputs.doubleToSz(Math.pow(10, -ninitval / 10.0)));
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

		numcols = 4;
		numrows = dremGOrecs.length;
		Object[][] tabledata = new Object[numrows][numcols];
		String[] columnNames = new String[numcols];

		columnNames[0] = "GO ID";
		columnNames[1] = "GO Category";
		columnNames[2] = "p-val overall";
		columnNames[3] = "p-val split";

		for (int nrow = 0; nrow < numrows; nrow++) {
			tabledata[nrow][0] = dremGOrecs[nrow].szID;
			tabledata[nrow][1] = dremGOrecs[nrow].szName;
			tabledata[nrow][2] = Util
					.doubleToSz(dremGOrecs[nrow].dminoverallpval);
			tabledata[nrow][3] = Util
					.doubleToSz(dremGOrecs[nrow].dminsplitpval);
		}

		hmst = new DREMGui_FilterGOModel(tabledata, columnNames);
		sorter = new TableSorter(hmst);
		final JTable table = new JTable(sorter);
		sorter.setTableHeader(table.getTableHeader());

		TableColumn column;
		column = table.getColumnModel().getColumn(0);
		column.setPreferredWidth(60);
		column = table.getColumnModel().getColumn(1);
		column.setPreferredWidth(180);
		column = table.getColumnModel().getColumn(2);
		column.setPreferredWidth(50);
		column = table.getColumnModel().getColumn(3);
		column.setPreferredWidth(50);

		// Create the scroll pane and add the table to it.
		JScrollPane scrollPane = new JScrollPane(table);
		table.setPreferredScrollableViewportSize(new Dimension(340, Math.min(
				(table.getRowHeight() + table.getRowMargin())
						* table.getRowCount(), 200)));

		table.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		rowSM = table.getSelectionModel();
		final DREMGui ftheDREMGui = theDREMGui;
		rowSM.addListSelectionListener(new ListSelectionListener() {
			public void valueChanged(ListSelectionEvent e) {
				// Ignore extra messages.
				if (e.getValueIsAdjusting())
					return;

				ListSelectionModel lsm = (ListSelectionModel) e.getSource();
				if (lsm.isSelectionEmpty()) {
					return;
				} else {
					int selectedRow = lsm.getMinSelectionIndex();
					szGO = (String) sorter.getValueAt(selectedRow, 0);
					ftheDREMGui.selectGO(szGO);
					addGOLabels(rootptr);
					setGOLabelTextVisible(
							rootptr,
							ftheDREMGui.bshowgolabels
									&& (ftheDREMGui.bglobalnode || !ftheDREMGui.battachlabels));

					hideButton.setEnabled(true);

				}

				unapplyButton.setEnabled(true);

			}
		});
		// Add the scroll pane to this panel.
		add(scrollPane);
		scrollPane.setAlignmentX(Component.LEFT_ALIGNMENT);

		JLabel theTopLabel = new JLabel(
				"  Only display GO enrichments with a p-value less than 10^-X where X is:");
		JPanel topPanel = new JPanel();
		topPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		topPanel.add(theTopLabel);
		topPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));

		topPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(topPanel);
		add(theSlider);
		theSlider.setAlignmentX(Component.LEFT_ALIGNMENT);

		JPanel labelPanel = new JPanel();
		labelPanel.add(pvalLabel);
		labelPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(labelPanel);
		labelPanel.setBackground(Color.white);
		labelPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

		JPanel countPanel = new JPanel();
		JLabel countLabel = new JLabel("counts should be based on ");
		countPanel.setBackground(Color.white);
		countPanel.add(countLabel);
		selectButton = new JRadioButton("Selected Genes");
		allButton = new JRadioButton("All Genes");
		pathButton = new JRadioButton("Overall Enrichments");
		splitButton = new JRadioButton("Split Enrichments");
		countGroup.add(selectButton);
		countGroup.add(allButton);
		selectButton.addItemListener(this);
		allButton.addItemListener(this);
		selectButton.setBackground(Color.white);
		allButton.setBackground(Color.white);
		countPanel.add(allButton);
		countPanel.add(selectButton);
		add(countPanel);
		if (ballgenes) {
			allButton.setSelected(true);
		} else {
			selectButton.setSelected(true);
		}
		countPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		countPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));

		JPanel typePanel = new JPanel();
		JLabel theGOLabel = new JLabel("p-values should be");
		typePanel.setBackground(Color.white);
		typePanel.add(theGOLabel);
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
		typePanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		typePanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		JPanel buttonPanel3 = new JPanel();
		buttonPanel3.setBackground(Color.white);
		hideButton = new JButton("Hide GO Labels");
		hideButton.setEnabled(false);
		hideButton.setActionCommand("hide");
		hideButton.addActionListener(this);

		colorButton = new JButton("Change Labels Color");
		colorButton.setActionCommand("color");
		colorButton.setMinimumSize(new Dimension(800, 20));
		colorButton.addActionListener(this);
		colorButton.setForeground(theDREMGui.goLabelColor);

		buttonPanel3.add(hideButton);
		buttonPanel3.add(colorButton);
		buttonPanel3.setAlignmentX(Component.LEFT_ALIGNMENT);

		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel3.add(helpButton);

		JPanel buttonPanel25 = new JPanel();
		buttonPanel25.setBackground(Color.white);
		buttonPanel25.setAlignmentX(Component.LEFT_ALIGNMENT);
		unapplyButton = new JButton("Unapply GO Selection Constraints");
		unapplyButton.setActionCommand("unapply");
		unapplyButton.addActionListener(this);
		if (tga.szSelectedGO == null) {
			unapplyButton.setEnabled(false);
		} else {
			unapplyButton.setEnabled(true);
		}

		buttonPanel25.add(unapplyButton);
		buttonPanel25.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		buttonPanel3.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(buttonPanel25);
		add(buttonPanel3);
	}

	/**
	 * Sets the color of the p-value significance labels
	 */
	void setGOLabelTextColor(DREM_Timeiohmm.Treenode treeptr, Color newColor) {
		if (treeptr != null) {
			treeptr.goText.setTextPaint(newColor);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setGOLabelTextColor(treeptr.nextptr[nchild], newColor);
			}
		}
	}

	/**
	 * Sets the visibility status of the GO labels
	 */
	void setGOLabelTextVisible(DREM_Timeiohmm.Treenode treeptr, boolean bvisible) {
		if (treeptr != null) {
			treeptr.goText.setVisible(bvisible);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setGOLabelTextVisible(treeptr.nextptr[nchild], bvisible);
			}
		}
	}

	/**
	 * Responds to movements in the slider specifying the significance threshold
	 */
	public void stateChanged(ChangeEvent e) {
		JSlider source = (JSlider) e.getSource();
		if (!source.getValueIsAdjusting()) {
			dgopval = Math.pow(10, -source.getValue() / 10.0);
			pvalLabel.setText("X = " + source.getValue() / 10.0
					+ "; p-value threshold is "
					+ DREMGui_KeyInputs.doubleToSz(dgopval));
			addGOLabels(rootptr);
		}
	}

	/**
	 * Responds to changes in the radio buttons
	 */
	public void itemStateChanged(ItemEvent e) {
		if (pathButton.isSelected()) {
			bsplitgolabels = false;
		} else {
			bsplitgolabels = true;
		}

		if (allButton.isSelected()) {
			ballgenes = true;
		} else {
			ballgenes = false;
		}

		addGOLabels(rootptr);
	}

	/**
	 * Responds to buttons being pressed
	 */
	public void actionPerformed(ActionEvent e) {

		String szcommand = e.getActionCommand();
		int nbadline = 0;

		if (szcommand.equals("color")) {

			Color newColor = JColorChooser.showDialog(this, "Choose Color",
					theDREMGui.goLabelColor);
			if (newColor != null) {
				theDREMGui.goLabelColor = newColor;
				colorButton.setForeground(newColor);
				setGOLabelTextColor(rootptr, newColor);
			}
		} else if (szcommand.equals("help")) {
			String szMessage = "This window allows one to reduce the set of genes currently displayed "
					+ "on the main interface to those that also belong to a certain GO category.  "
					+ "The GO category is selected by clicking on a row of the table. To change "
					+ "the GO category one simply needs to click on a different row of the the table.  "
					+ "To no longer select genes by any GO category press the 'Unapply GO Selection "
					+ "Constraints' button. When genes are selected by a GO category, "
					+ "significant p-values appear on the map to the immediate right of nodes on "
					+ "the map. The threshold for significant p-values is defined based on the "
					+ "value on the slider. Let X be the value of the slider then 10^(-X) is the p-value "
					+ "threshold. The 'counts should be based on' can be set to 'All Genes' or 'Selected Genes'. "
					+ "Under the 'All Genes' options the counts and enrichments calculations consider all "
					+ "genes going through the path. Under the 'Selected Genes' "
					+ "option counts and enrichments calculations consider only the set of genes going "
					+ "through the path and meeting the other selection constraints "
					+ "(Selection by TF and Gene Set). There is also the option 'p-values should be', which "
					+ "can be 'Overall Enrichments' or 'Split Enrichments'. Overall enrichments "
					+ "compute p-value where the base set of "
					+ "genes is all genes in the expression data file or the "
					+ "'Pre-filtered Gene File'. Split enrichments are based on just the "
					+ "genes assigned to the prior split. Pressing the 'Hide Labels' button "
					+ "hides these labels on the map. To change the colors of these labels press "
					+ "the 'Change Labels' button. The color of the text of this button will match the "
					+ "color of the GO labels on the map.";

			Util.renderDialog(theFrame, szMessage, -300, -100);
		} else if (szcommand.equals("hide")) {
			if (theDREMGui.bshowgolabels) {
				theDREMGui.bshowgolabels = false;
				hideButton.setText("Show GO Labels");
				setGOLabelTextVisible(rootptr, false);
			} else {
				theDREMGui.bshowgolabels = true;
				hideButton.setText("Hide GO Labels");
				setGOLabelTextVisible(rootptr, true);
			}

		} else if (szcommand.equals("unapply")) {
			theDREMGui.bapplygolabels = false;
			unapplyButton.setEnabled(false);
			hideButton.setEnabled(false);
			theDREMGui.unselectGO();
			rowSM.clearSelection();
			setGOLabelTextVisible(rootptr, false);
		}
	}

	/**
	 * Adds p-value labels meeting the significance threshold to the interface
	 */
	public void addGOLabels(DREM_Timeiohmm.Treenode treeptr) {
		if (treeptr != null) {
			int nrow = rowSM.getMinSelectionIndex();

			if (nrow >= 0) {
				theDREMGui.bapplygolabels = true;
				int nmatch = 0;
				int ncandidate = 0;
				double dpval;
				HashSet hsGO;
				GoAnnotations tga = theDREMGui.theTimeiohmm.theDataSet.tga;

				String szGO = (String) sorter.getValueAt(nrow, 0);

				if (!bsplitgolabels) {
					for (int ngene = 0; ngene < treeptr.bInNode.length; ngene++) {
						if (treeptr.bInNode[ngene]
								&& (ballgenes || theDREMGui.bTFVisible[ngene]
										&& theDREMGui.bSetVisible[ngene])) {
							ncandidate++;
							if (theDREMGui.bGOVisible[ngene]) {
								nmatch++;
							}
						}
					}

					int ncategoryall = ((Integer) tga.htFullCount.get(szGO))
							.intValue();
					int nval = nmatch - 1;

					dpval = StatUtil.hypergeometrictail(nval, ncategoryall,
							tga.numtotalgenes - ncategoryall, ncandidate);

					if (dpval <= dgopval) {
						treeptr.szgolabel = nmatch + ";"
								+ Util.doubleToSz(dpval);
					} else {
						treeptr.szgolabel = "";
					}
				} else if ((treeptr.parent != null)
						&& (treeptr.parent.numchildren >= 2)) {
					int nparentbase = 0;
					int nparentcategory = 0;

					for (int ngene = 0; ngene < treeptr.bInNode.length; ngene++) {
						if (treeptr.parent.bInNode[ngene]) {
							nparentbase++;
							if (theDREMGui.bGOVisible[ngene]) {
								nparentcategory++;
							}

							if (ballgenes
									|| (theDREMGui.bTFVisible[ngene] && theDREMGui.bSetVisible[ngene])) {
								if (treeptr.bInNode[ngene]) {
									ncandidate++;
									if (theDREMGui.bGOVisible[ngene]) {
										nmatch++;
									}
								}
							}
						}
					}

					int nval = nmatch - 1;

					dpval = StatUtil.hypergeometrictail(nval, nparentcategory,
							nparentbase - nparentcategory, ncandidate);
					if (dpval <= dgopval) {
						treeptr.szgolabel = nmatch + ";"
								+ Util.doubleToSz(dpval);
					} else {
						treeptr.szgolabel = "";
					}
				} else {
					treeptr.szgolabel = "";
				}

				treeptr.goText.setText(treeptr.szgolabel);
				for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
					addGOLabels(treeptr.nextptr[nchild]);
				}
			}
		}
	}
}