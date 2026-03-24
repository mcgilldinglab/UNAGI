package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.text.NumberFormat;

/**
 * Class encapsulates the interface option window for the main interface
 */
public class DREMGui_InterfaceOptions extends JPanel implements ChangeListener,
		ItemListener, ActionListener {
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	JSlider theSlider;
	JSlider theSliderX;
	JSlider theColorSlider;
	JSlider theNodeSlider;
	JFrame theFrame;

	JButton hideButton;
	Hashtable theDictionary;
	DREMGui theDREMGui;
	JLabel pvalLabel;
	JLabel pvalLabelX;
	JLabel nodeLabel;

	JCheckBox attachBox;
	ButtonGroup group = new ButtonGroup();
	JRadioButton uniformButton;
	JRadioButton realButton;
	ButtonGroup autogroup = new ButtonGroup();
	JRadioButton holdButton;
	JRadioButton autoButton;

	/**
	 * Constructs the inteface option dialog window
	 */
	public DREMGui_InterfaceOptions(JFrame theFrame, DREMGui theDREMGui) {
		this.theFrame = theFrame;

		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);

		JPanel colorPanel = new JPanel();
		JLabel colorLabel = new JLabel("Gene colors should be based on edge:");
		colorPanel.setBackground(new Color((float) 0.0, (float) 1.0,
				(float) 0.0, (float) 0.4));
		colorPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		colorPanel.add(colorLabel);
		add(colorPanel);

		int npoints = theDREMGui.theTimeiohmm.theDataSet.data[0].length - 1;
		theColorSlider = new JSlider(0, npoints, theDREMGui.ncolortime);
		theDictionary = new Hashtable();
		theDictionary.put(new Integer(0), new JLabel("Rand."));
		for (int nindex = 1; nindex <= npoints; nindex++) {
			theDictionary.put(new Integer(nindex), new JLabel("" + nindex));
		}
		theColorSlider.setLabelTable(theDictionary);
		theColorSlider.setMinimumSize(new Dimension(800, 600));
		theColorSlider.setMajorTickSpacing(1);
		theColorSlider.setSnapToTicks(true);

		theColorSlider.setPaintTicks(true);
		theColorSlider.setPaintLabels(true);
		theColorSlider.addChangeListener(this);
		theColorSlider.setPaintTicks(true);
		theColorSlider.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(theColorSlider);

		autogroup = new ButtonGroup();
		holdButton = new JRadioButton("Hold Fixed");
		autoButton = new JRadioButton("Automatically Adjust");
		autogroup.add(holdButton);
		autogroup.add(autoButton);
		if (theDREMGui.bholdedge) {
			holdButton.setSelected(true);
		} else {
			autoButton.setSelected(true);
		}
		holdButton.setBackground(Color.white);
		autoButton.setBackground(Color.white);
		holdButton.addItemListener(this);
		autoButton.addItemListener(this);
		add(holdButton);
		add(autoButton);

		JLabel theTopLabel = new JLabel(
				"             Scale Y-axis by the factor:              ");
		JPanel topPanel = new JPanel();
		topPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		topPanel.add(theTopLabel);
		topPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));
		add(topPanel);

		theSlider = new JSlider(0, 100, (int) (theDREMGui.dscaley * 10));
		theDictionary = new Hashtable();
		for (int nindex = 0; nindex <= 10; nindex++) {
			theDictionary
					.put(new Integer(nindex * 10), new JLabel("" + nindex));
		}
		theSlider.setLabelTable(theDictionary);
		theSlider.setMinimumSize(new Dimension(800, 600));
		this.theDREMGui = theDREMGui;
		theSlider.setMajorTickSpacing(10);
		theSlider.setMinorTickSpacing(2);
		theSlider.setPaintTicks(true);
		theSlider.setPaintLabels(true);
		theSlider.addChangeListener(this);

		theSlider.setPaintTicks(true);
		theSlider.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(theSlider);

		JPanel labelPanel = new JPanel();
		pvalLabel = new JLabel("Y-axis scale factor is "
				+ doubleToSz(theDREMGui.dscaley));
		labelPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		labelPanel.add(pvalLabel);
		labelPanel.setBackground(Color.white);
		add(labelPanel);

		JLabel theTopLabelX = new JLabel(
				"             Scale X-axis by the factor:              ");
		JPanel topPanelX = new JPanel();
		topPanelX.setAlignmentX(Component.LEFT_ALIGNMENT);
		topPanelX.add(theTopLabelX);
		topPanelX.setBackground(new Color((float) 0.0, (float) 1.0,
				(float) 0.0, (float) 0.4));
		add(topPanelX);

		theSliderX = new JSlider(0, 100, (int) (theDREMGui.dscalex * 10));
		theSliderX.setLabelTable(theDictionary);
		theSliderX.setMinimumSize(new Dimension(800, 600));
		theSliderX.setMajorTickSpacing(10);
		theSliderX.setMinorTickSpacing(2);
		theSliderX.setPaintTicks(true);
		theSliderX.setPaintLabels(true);
		theSliderX.addChangeListener(this);
		theSliderX.setPaintTicks(true);
		theSliderX.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(theSliderX);

		JPanel labelPanelX = new JPanel();
		pvalLabelX = new JLabel("X-axis scale factor is "
				+ doubleToSz(theDREMGui.dscalex));
		labelPanelX.setAlignmentX(Component.LEFT_ALIGNMENT);
		labelPanelX.add(pvalLabelX);
		labelPanelX.setBackground(Color.white);
		add(labelPanelX);

		JPanel binPanel = new JPanel();
		binPanel.setBackground(new Color((float) 0.0, (float) 1.0, (float) 0.0,
				(float) 0.4));
		add(binPanel);

		JLabel theXLabel = new JLabel(
				"           X-axis scale should be:            ");
		add(theXLabel);

		group = new ButtonGroup();
		realButton = new JRadioButton("Based on Real Time");
		uniformButton = new JRadioButton("Uniform");
		group.add(realButton);
		group.add(uniformButton);
		if (theDREMGui.brealXaxis) {
			realButton.setSelected(true);
		} else {
			uniformButton.setSelected(true);
		}

		if (theDREMGui.binvalidreal) {
			realButton.setEnabled(false);
		}

		uniformButton.setBackground(Color.white);
		realButton.setBackground(Color.white);
		uniformButton.addItemListener(this);
		realButton.addItemListener(this);
		add(uniformButton);
		add(realButton);

		JLabel theNodeLabel = new JLabel(
				"             Scale node areas by the factor:              ");
		JPanel nodeLabelPanel = new JPanel();
		nodeLabelPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		nodeLabelPanel.add(theNodeLabel);
		nodeLabelPanel.setBackground(new Color((float) 0.0, (float) 1.0,
				(float) 0.0, (float) 0.4));
		add(nodeLabelPanel);
		theNodeSlider = new JSlider(0, 50, (int) (theDREMGui.dnodek * 10));

		theNodeSlider.setLabelTable(theDictionary);
		theNodeSlider.setMinimumSize(new Dimension(800, 600));
		theNodeSlider.setMajorTickSpacing(10);
		theNodeSlider.setMinorTickSpacing(1);
		theNodeSlider.setPaintTicks(true);
		theNodeSlider.setPaintLabels(true);
		theNodeSlider.addChangeListener(this);
		theNodeSlider.setPaintTicks(true);
		theNodeSlider.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(theNodeSlider);

		JPanel nodelabelPanel = new JPanel();
		nodeLabel = new JLabel("Node area scale factor is "
				+ doubleToSz(theDREMGui.dnodek));
		nodelabelPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		nodelabelPanel.add(nodeLabel);
		nodelabelPanel.setBackground(Color.white);
		add(nodelabelPanel);

		JPanel breakPanel = new JPanel();
		breakPanel.setBackground(new Color((float) 0.0, (float) 1.0,
				(float) 0.0, (float) 0.4));
		add(breakPanel);

		attachBox = new JCheckBox("Hide All Labels When Hiding Nodes");
		attachBox.setBackground(Color.white);
		attachBox.setSelected(theDREMGui.battachlabels);
		attachBox.addActionListener(this);

		JPanel attachPanel = new JPanel();
		attachPanel.setBackground(Color.white);
		attachPanel.add(attachBox);
		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		attachPanel.add(helpButton);
		attachPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		attachPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
		add(attachPanel);
	}

	/**
	 * Responds to actions on the interface window
	 */
	public void actionPerformed(ActionEvent e) {
		theDREMGui.battachlabels = attachBox.isSelected();
		String szcommand = e.getActionCommand();

		if (szcommand.equals("help")) {
			String szMessage = "The first option is 'Gene colors should be based on edge' determines the color of "
					+ "the time series on the main interface. By default "
					+ "all time series have random colors. If this parameter is set to 1, "
					+ "then the time series of the genes have the same "
					+ "color as the edge between time point 0 and the next time point of the "
					+ "path on the DREM map to which the genes "
					+ "were assigned. In general if the parameter is set to i every "
					+ "time series has the same color as the ith edge of the "
					+ "path to which it is assigned in the DREM map. The next option "
					+ "determines whether DREM should 'Hold Fixed' "
					+ "the Gene colors should be based on edge parameter value or "
					+ "'Automatically Adjust' it based on the edge or a node "
					+ "of the DREM map a user clicked. If 'Automatically Adjust' is "
					+ "selected the value of the parameter will be set to "
					+ "correspond to the node or edge the user clicked on.  "
					+ "The next two options,' Scale Y-axis by the factor' and "
					+ "'Scale X-axis by the factor', allow one to adjust the y-scale "
					+ "and x-scale of the main window. The default scale for the x and y-axes "
					+ "are multiplied proportional to the value of this parameter.  "
					+ "The 'X-axis scale should be' option can either be set to 'Uniform' in "
					+ "which case each time point is uniformly "
					+ "spaced on the screen independent of the real sampling rate "
					+ "or it can be 'Based on Real Time' in which case the "
					+ "spacing of time points is based proportional to the sampling rate.  "
					+ "The 'Scale node areas by the factor' slider allows a user to scale the "
					+ "area of the nodes on the main interface proportional to the value of "
					+ "this parameter. Each individual node will continue to have an area proportional to "
					+ "the standard deviation of the distribution of genes associated with it.  "
					+ "The final option 'Hide All Labels When Hiding Nodes' determines if the labels are "
					+ "also hidden when a user presses the 'Hide Nodes' button on the main interface. "
					+ "If the box is not checked then just the nodes and edges will "
					+ "be hidden, but not the labels.";
			Util.renderDialog(theFrame, szMessage, -350, -100);
		}

		if ((theDREMGui.battachlabels) && (!theDREMGui.bglobalnode)) {
			theDREMGui.hidelabels();
		} else if (!theDREMGui.battachlabels) {
			theDREMGui.showlabels();
		}
	}

	/**
	 * Responds to item changes on the interface window
	 */
	public void itemStateChanged(ItemEvent e) {
		if (realButton.isSelected()) {
			theDREMGui.brealXaxis = true;
		} else {
			theDREMGui.brealXaxis = false;
		}

		if (holdButton.isSelected()) {
			theDREMGui.bholdedge = true;
		} else {
			theDREMGui.bholdedge = false;
		}

		theDREMGui.drawmain();
	}

	/**
	 * Responds to state changes on the interface window
	 */
	public void stateChanged(ChangeEvent e) {
		JSlider source = (JSlider) e.getSource();
		if (!source.getValueIsAdjusting()) {
			if (source == theSlider) {
				theDREMGui.dscaley = source.getValue() / 10.0;
				pvalLabel.setText("Y-axis scale factor is "
						+ doubleToSz(theDREMGui.dscaley));
				theDREMGui.drawmain();
			} else if (source == theSliderX) {
				theDREMGui.dscalex = source.getValue() / 10.0;
				pvalLabelX.setText("X-axis scale factor is "
						+ doubleToSz(theDREMGui.dscalex));
				theDREMGui.drawmain();
			} else if (source == theNodeSlider) {
				theDREMGui.dnodek = source.getValue() / 10.0;

				nodeLabel.setText("Node area scale factor "
						+ doubleToSz(theDREMGui.dnodek));
				theDREMGui.drawmain();
			} else if (source == theColorSlider) {
				theDREMGui.ncolortime = source.getValue();
				theDREMGui.setGeneColors();
			}
		}
	}

	// /////////////////////////////////////////////
	/**
	 * Converts a dobule value to a decimal formatted string
	 */
	public static String doubleToSz(double dval) {
		String szexp;
		NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
		nf2.setMinimumFractionDigits(1);
		nf2.setMaximumFractionDigits(1);

		szexp = nf2.format(dval);

		return szexp;

	}

}