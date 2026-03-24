package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import edu.umd.cs.piccolo.PNode;
import edu.umd.cs.piccolo.nodes.PPath;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.text.NumberFormat;

/**
 * Class encapsulates the window used to specify the criteria that determines
 * transcription factor labels appear on the main interface
 */
public class DREMGui_KeyInputs extends JPanel implements ActionListener,
		ChangeListener, ItemListener {

	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	JSlider theSlider;
	JSlider miRNASlider;
	JSlider theSliderPercent;
	JButton hideButton;
	JButton regToggleButton;
	JButton regColorToggleButton;
	JButton saveButton;
	JFrame saveRegsFrame;
	boolean displayTF = true;
	boolean displayMIRNA = true;
	Hashtable<Integer, JLabel> theDictionary;
	Hashtable<Integer, JLabel> miRNADictionary;
	Hashtable<Integer, JLabel> theDictionaryPercent;
	DREMGui theDREMGui;
	JLabel pvalLabel;
	JLabel pvalmiRNALabel;
	JLabel percentLabel;

	ButtonGroup group = new ButtonGroup();
	JRadioButton splitButton;
	JRadioButton edgeSplitButton;
	JRadioButton edgeFullButton;
	JRadioButton activityButton;
	JRadioButton firstActivityButton;
	JFrame theFrame;

	NumberFormat nf2;

	/**
	 * Class constructor - builds interface window
	 */
	public DREMGui_KeyInputs(JFrame theFrametemp, DREMGui theDREMGuitemp) {
		this.theFrame = theFrametemp;
		this.theDREMGui = theDREMGuitemp;
		nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(2);
		nf2.setMaximumFractionDigits(2);
		boolean mirnapresent = theDREMGui.theTimeiohmm.bindingData.existingRegTypes
				.contains(RegulatorBindingData.MIRNA);

		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);
		int ninitval = (int) Math.round(-100
				* Math.log(theDREMGui.dkeyinputpvalue) / Math.log(10)) / 10;

		// pval = 10^-X
		// log p-val/log 10 = -X
		// X = -log p-val/log 10
		theDREMGui.dkeyinputpvalue = Math.pow(10, -ninitval / 10.0);
		theDREMGui.dkeymiRNAinputpvalue = Math.pow(10, -ninitval / 10.0);

		JLabel theTopLabel = new JLabel(
				"  Only display key TFs with a score less than 10^-X where X is:");
		JPanel topPanel = new JPanel();
		topPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		topPanel.add(theTopLabel);
		topPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));

		add(topPanel);
		theSlider = new JSlider(0, 120, ninitval);
		theDictionary = new Hashtable<Integer, JLabel>();
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
		theSlider.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(theSlider);

		JPanel labelPanel = new JPanel();
		pvalLabel = new JLabel("X = " + ninitval / 10.0
				+ ";  score threshold is "
				+ doubleToSz(Math.pow(10, -ninitval / 10.0)));
		labelPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		labelPanel.add(pvalLabel);
		labelPanel.setBackground(Color.white);
		add(labelPanel);

		JLabel miRNATopLabel = new JLabel(
				"  Only display key miRNAs with a score less than 10^-X where X is:");
		JPanel topmiRNAPanel = new JPanel();
		topmiRNAPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		topmiRNAPanel.add(miRNATopLabel);
		topmiRNAPanel.setBackground(new Color((float) 0.0, (float) 1.0,
				(float) 0.0, (float) 0.4));
		if (mirnapresent)
			add(topmiRNAPanel);
		miRNASlider = new JSlider(0, 120, ninitval);
		miRNADictionary = new Hashtable<Integer, JLabel>();
		for (int nindex = 0; nindex <= 12; nindex++) {
			miRNADictionary.put(new Integer(nindex * 10), new JLabel(""
					+ nindex));
		}
		miRNASlider.setLabelTable(miRNADictionary);
		miRNASlider.setMajorTickSpacing(10);
		miRNASlider.setMinorTickSpacing(5);
		miRNASlider.setPaintTicks(true);
		miRNASlider.setPaintLabels(true);
		miRNASlider.addChangeListener(this);
		miRNASlider.setPaintTicks(true);
		miRNASlider.setAlignmentX(Component.LEFT_ALIGNMENT);
		if (mirnapresent)
			add(miRNASlider);

		JPanel miRNALabelPanel = new JPanel();
		pvalmiRNALabel = new JLabel("X = " + ninitval / 10.0
				+ ";  score threshold is "
				+ doubleToSz(Math.pow(10, -ninitval / 10.0)));
		miRNALabelPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		miRNALabelPanel.add(pvalmiRNALabel);
		miRNALabelPanel.setBackground(Color.white);
		if (mirnapresent)
			add(miRNALabelPanel);

		JLabel binLabel;
		binLabel = new JLabel(
				" Compute and display key regulator significance based on: ");

		JPanel binPanel = new JPanel();
		binPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));
		binPanel.add(binLabel);
		binPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(binPanel);

		binLabel.setAlignmentX(Component.LEFT_ALIGNMENT);
		theDREMGui.blowbelow = true;
		theDREMGui.blowabove = true;

		splitButton = new JRadioButton("Split Significance");
		edgeSplitButton = new JRadioButton(
				"Path Significance Conditional on Split");
		edgeFullButton = new JRadioButton("Path Significance Overall");
		activityButton = new JRadioButton("Activity Score");
		firstActivityButton = new JRadioButton(
				"Activity Score (first appearance per path)");

		group = new ButtonGroup();

		if (theDREMGui.nKeyInputType == 0) {
			splitButton.setSelected(true);
		} else if (theDREMGui.nKeyInputType == 1) {
			edgeSplitButton.setSelected(true);
		} else if (theDREMGui.nKeyInputType == 778) {
			firstActivityButton.setSelected(true);
		} else if (theDREMGui.nKeyInputType == 777) {
			activityButton.setSelected(true);
		} else // (theDREMGui.nKeyInputType == 2)
		{
			edgeFullButton.setSelected(true);
		}
		group.add(edgeSplitButton);
		group.add(edgeFullButton);
		group.add(splitButton);
		if(theDREMGui.theTimeiohmm.bindingData.regPriors != null)
		{
			group.add(activityButton);
			group.add(firstActivityButton);
		}

		splitButton.addItemListener(this);
		edgeSplitButton.addItemListener(this);
		edgeFullButton.addItemListener(this);
		activityButton.addItemListener(this);
		firstActivityButton.addItemListener(this);

		add(edgeSplitButton);
		add(edgeFullButton);
		add(splitButton);
		if(theDREMGui.theTimeiohmm.bindingData.regPriors != null)
		{
			add(activityButton);
			add(firstActivityButton);
		}

		splitButton.setAlignmentX(Component.LEFT_ALIGNMENT);
		splitButton.setBackground(Color.white);
		edgeSplitButton.setAlignmentX(Component.LEFT_ALIGNMENT);
		edgeSplitButton.setBackground(Color.white);
		edgeFullButton.setAlignmentX(Component.LEFT_ALIGNMENT);
		edgeFullButton.setBackground(Color.white);
		activityButton.setAlignmentX(Component.LEFT_ALIGNMENT);
		activityButton.setBackground(Color.white);
		firstActivityButton.setAlignmentX(Component.LEFT_ALIGNMENT);
		firstActivityButton.setBackground(Color.white);

		JLabel theTopLabel2 = new JLabel(
				"For Path Significance Conditional on Split - Minimum Split %: ");
		JPanel topPanel2 = new JPanel();
		topPanel2.setAlignmentX(Component.LEFT_ALIGNMENT);
		topPanel2.add(theTopLabel2);
		topPanel2.setBackground(new Color((float) 0.0, (float) 1.0,
				(float) 0.0, (float) 0.4));
		add(topPanel2);

		if (DREM_Timeiohmm.BDEBUG) {
			System.out.println("!!!!" + theDREMGui.dsplitpercent);
		}
		int ninitvalpercent = (int) Math.round(theDREMGui.dsplitpercent * 100);

		// pval = 10^-X
		// log p-val/log 10 = -X
		// X = -log p-val/log 10
		theDREMGui.dsplitpercent = ninitvalpercent / 100.0;

		theSliderPercent = new JSlider(0, 100, ninitvalpercent);
		theDictionaryPercent = new Hashtable<Integer, JLabel>();
		for (int nindex = 0; nindex <= 10; nindex++) {
			theDictionaryPercent.put(new Integer(nindex * 10), new JLabel(""
					+ nindex * 10));
		}
		theSliderPercent.setLabelTable(theDictionaryPercent);
		theSliderPercent.setMajorTickSpacing(10);
		theSliderPercent.setMinorTickSpacing(5);
		theSliderPercent.setPaintTicks(true);
		theSliderPercent.setPaintLabels(true);
		theSliderPercent.addChangeListener(this);
		theSliderPercent.setPaintTicks(true);
		theSliderPercent.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(theSliderPercent);

		JPanel labelPanel2 = new JPanel();
		percentLabel = new JLabel(" Minimum Split % is "
				+ nf2.format(100 * theDREMGui.dsplitpercent));
		labelPanel2.setAlignmentX(Component.LEFT_ALIGNMENT);
		labelPanel2.add(percentLabel);
		labelPanel2.setBackground(Color.white);
		labelPanel2.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		add(labelPanel2);

		hideButton = new JButton("Hide Key Reg Labels");
		hideButton.setActionCommand("hide");
		hideButton.addActionListener(this);

		regToggleButton = new JButton("Toggle to Only TF");
		regToggleButton.setActionCommand("toggle");
		regToggleButton.setMinimumSize(new Dimension(800, 20));
		regToggleButton.addActionListener(this);
		regToggleButton.setForeground(theDREMGui.keyInputLabelColor);

		regColorToggleButton = new JButton("Toggle Exp. Coloring");
		regColorToggleButton.setActionCommand("togglecolor");
		regColorToggleButton.setMinimumSize(new Dimension(800, 20));
		regColorToggleButton.addActionListener(this);
		regColorToggleButton.setForeground(theDREMGui.keyInputLabelColor);
		
		JPanel buttonPanel = new JPanel();

		buttonPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		buttonPanel.setBackground(Color.white);

		buttonPanel.add(hideButton);
		if (mirnapresent)
			buttonPanel.add(regToggleButton);
		buttonPanel.add(regColorToggleButton);
		
		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel.add(helpButton);
		add(buttonPanel);

		JPanel savePanel = new JPanel();
		savePanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		savePanel.setBackground(Color.white);

		saveButton = new JButton("Save Significant Regulators to File");
		saveButton.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent event) {
				javax.swing.SwingUtilities.invokeLater(new Runnable() {
					public void run() {
						if (saveRegsFrame == null) {
							saveRegsFrame = new JFrame(
									"Save Regulators to File");
							saveRegsFrame
									.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
							saveRegsFrame.setLocation(400, 300);
							DREMGui_SaveSigRegs newContentPane = new DREMGui_SaveSigRegs(
									theDREMGui, saveRegsFrame);
							newContentPane.setOpaque(true);
							// content panes must be opaque
							saveRegsFrame.setContentPane(newContentPane);
							// Display the window.
							saveRegsFrame.pack();
						} else {
							saveRegsFrame.setExtendedState(Frame.NORMAL);
						}
						saveRegsFrame.setVisible(true);
					}
				});
			}
		});

		savePanel.add(saveButton);
		add(savePanel);
	}

	/**
	 * Responds to changes in the significance threshold at which transcription
	 * factor labels should appears
	 */
	public void stateChanged(ChangeEvent e) {
		JSlider source = (JSlider) e.getSource();

		theDREMGui.sigRegs = new HashSet<String>();
		if (!source.getValueIsAdjusting()) {
			if (source == theSlider) {
				theDREMGui.dkeyinputpvalue = Math.pow(10,
						-source.getValue() / 10.0);
				pvalLabel.setText("X = " + source.getValue() / 10.0
						+ "; score threshold is "
						+ doubleToSz(theDREMGui.dkeyinputpvalue));
				int nsize = theDREMGui.hidesigList.size();
				for (int nindex = 0; nindex < nsize; nindex++) {
					DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
							.get(nindex);
					theSigInfoRec.border.removeAllChildren();
					theDREMGui.setSigText(theSigInfoRec.theSigTF,
							(PPath) theSigInfoRec.border, theSigInfoRec.ntype,
							theSigInfoRec.ndepth, displayTF, displayMIRNA,
							theSigInfoRec.node, theDREMGui.colorSigTFs);
				}
			} else if (source == theSliderPercent) {
				theDREMGui.dsplitpercent = source.getValue() / 100.0;
				percentLabel.setText(" Minimum Split % is "
						+ nf2.format(100 * theDREMGui.dsplitpercent));
				int nsize = theDREMGui.hidesigList.size();
				for (int nindex = 0; nindex < nsize; nindex++) {
					DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
							.get(nindex);
					if (theSigInfoRec.ntype == 1) // split path type
					{
						theSigInfoRec.border.removeAllChildren();
						theDREMGui.setSigText(theSigInfoRec.theSigTF,
								(PPath) theSigInfoRec.border,
								theSigInfoRec.ntype, theSigInfoRec.ndepth,
								displayTF, displayMIRNA, theSigInfoRec.node, theDREMGui.colorSigTFs);
					}
				}
			} else if (source == miRNASlider) {
				theDREMGui.dkeymiRNAinputpvalue = Math.pow(10, -source
						.getValue() / 10.0);
				pvalmiRNALabel.setText("X = " + source.getValue() / 10.0
						+ "; score threshold is "
						+ doubleToSz(theDREMGui.dkeymiRNAinputpvalue));
				int nsize = theDREMGui.hidesigList.size();
				for (int nindex = 0; nindex < nsize; nindex++) {
					DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
							.get(nindex);
					theSigInfoRec.border.removeAllChildren();
					theDREMGui.setSigText(theSigInfoRec.theSigTF,
							(PPath) theSigInfoRec.border, theSigInfoRec.ntype,
							theSigInfoRec.ndepth, displayTF, displayMIRNA, 
							theSigInfoRec.node, theDREMGui.colorSigTFs);
				}
			}
		}
	}

	/**
	 * Converts a double value to a string as formatted as displayed on this
	 * interface
	 */
	public static String doubleToSz(double dval) {
		String szexp;
		double dtempval = dval;
		int nexp = 0;

		NumberFormat nf3 = NumberFormat.getInstance(Locale.ENGLISH);
		nf3.setMinimumFractionDigits(3);
		nf3.setMaximumFractionDigits(3);

		NumberFormat nf4 = NumberFormat.getInstance(Locale.ENGLISH);
		nf4.setMinimumFractionDigits(2);
		nf4.setMaximumFractionDigits(2);

		if (dval <= 0) {
			szexp = "0.000";
		} else {
			while ((dtempval < 0.9995) && (dtempval > 0)) {
				nexp--;
				dtempval = dtempval * 10;
			}
			dtempval = dval * Math.pow(10, -nexp);
			if (nexp < -3)
				szexp = nf4.format(dtempval) + " * 10^" + nexp;
			else
				szexp = nf3.format(dval);
		}

		return szexp;
	}

	/**
	 * Responds to changes with the radio buttons
	 */
	public void itemStateChanged(ItemEvent e) {
		if (splitButton.isSelected()) {
			theDREMGui.nKeyInputType = 0;
		} else if (edgeSplitButton.isSelected()) {
			theDREMGui.nKeyInputType = 1;
		} else if (edgeFullButton.isSelected()) {
			theDREMGui.nKeyInputType = 2;
		} else if (activityButton.isSelected()) {
			theDREMGui.nKeyInputType = 777;
		} else if (firstActivityButton.isSelected()) {
			theDREMGui.nKeyInputType = 778;
		}

		int nsize = theDREMGui.hidesigList.size();

		theDREMGui.sigRegs = new HashSet<String>();
		for (int nindex = 0; nindex < nsize; nindex++) {
			DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
					.get(nindex);

			theSigInfoRec.border.removeAllChildren();
			theDREMGui.setSigText(theSigInfoRec.theSigTF,
					(PPath) theSigInfoRec.border, theSigInfoRec.ntype,
					theSigInfoRec.ndepth, displayTF, displayMIRNA, 
					theSigInfoRec.node, theDREMGui.colorSigTFs);
		}
	}

	/**
	 * Responds to buttons being pressed on the interface window
	 */
	public void actionPerformed(ActionEvent e) {
		String szcommand = e.getActionCommand();

		if (szcommand.equals("help")) {
			String szMessage = "This window controls the Key Input labels appearing on the DREM output interface.   "
					+ "Consult section 4.4 of the user manual for more details on this window. "
					+ "TF activity scores do not use pvalues so the TFs shown will be those with "
					+ "activity scores greater than or equal to the inverse of the threshold.";
			Util.renderDialog(theFrame, szMessage, -350, -100);
		} else if (szcommand.equals("toggle")) {
			theDREMGui.sigRegs = new HashSet<String>();
			
			if (displayTF && displayMIRNA) {
				displayMIRNA = false;
				regToggleButton.setText("Toggle to Only miRNA");
			} else if (displayTF) {
				displayTF = false;
				displayMIRNA = true;
				regToggleButton.setText("Toggle to miRNA and TF");
			} else// displayMIRNA == true
			{
				displayTF = true;
				regToggleButton.setText("Toggle to Only TF");
			}
			miRNASlider.setEnabled(displayMIRNA);
			theSlider.setEnabled(displayTF);
			int nsize = theDREMGui.hidesigList.size();
			for (int nindex = 0; nindex < nsize; nindex++) {
				DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
						.get(nindex);

				theSigInfoRec.border.removeAllChildren();
				theDREMGui.setSigText(theSigInfoRec.theSigTF,
						(PPath) theSigInfoRec.border, theSigInfoRec.ntype,
						theSigInfoRec.ndepth, displayTF, displayMIRNA, 
						theSigInfoRec.node, theDREMGui.colorSigTFs);
			}
			theFrame.pack();
		} else if (szcommand.equals("togglecolor")) {
			theDREMGui.colorSigTFs = !theDREMGui.colorSigTFs;
			
			int nsize = theDREMGui.hidesigList.size();
			for (int nindex = 0; nindex < nsize; nindex++) {
				DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
						.get(nindex);

				theSigInfoRec.border.removeAllChildren();
				theDREMGui.setSigText(theSigInfoRec.theSigTF,
						(PPath) theSigInfoRec.border, theSigInfoRec.ntype,
						theSigInfoRec.ndepth, displayTF, displayMIRNA, 
						theSigInfoRec.node, theDREMGui.colorSigTFs);
			}
		} else if (szcommand.equals("hide")) {
			int nsize = theDREMGui.hidesigList.size();

			if (theDREMGui.bshowkeyinputs) {
				hideButton.setText("Show Key Reg Labels");
				theDREMGui.bshowkeyinputs = false;

				for (int nindex = 0; nindex < nsize; nindex++) {
					DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
							.get(nindex);

					PPath rect = (PPath) theSigInfoRec.border;

					rect.setVisible(false);
					rect.setPickable(false);
					for (Object osmallrect : rect.getChildrenReference()) {
						PNode smallrect = ((PNode) osmallrect);
						smallrect.setVisible(false);
						for (Object otext : smallrect.getChildrenReference()) {
							PNode text = (PNode) otext;
							text.setVisible(false);
						}
					}
				}
			} else {
				hideButton.setText("Hide Key Reg Labels");
				theDREMGui.bshowkeyinputs = true;
				for (int nindex = 0; nindex < nsize; nindex++) {
					DREMGui.SigInfoRec theSigInfoRec = (DREMGui.SigInfoRec) theDREMGui.hidesigList
							.get(nindex);
					PPath rect = (PPath) theSigInfoRec.border;
					rect.setVisible(true);
					rect.setPickable(true);
					// If there is no rectangle we don't want to show the line
					if (rect.getWidth() != 0) {
						for (Object osmallrect : rect.getChildrenReference()) {
							PNode smallrect = ((PNode) osmallrect);
							smallrect.setVisible(true);
							for (Object otext : smallrect
									.getChildrenReference()) {
								PNode text = (PNode) otext;
								text.setVisible(true);
							}
						}
					}
				}
			}
			theFrame.pack();
		}
	}
}