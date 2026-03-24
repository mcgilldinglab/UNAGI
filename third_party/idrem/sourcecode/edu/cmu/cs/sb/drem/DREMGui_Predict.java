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
import javax.swing.border.*;

/**
 * Class encapsulates a window to specify an input for each transcription
 * factor, then displays a probability of transitioning to each hidden state
 */
public class DREMGui_Predict extends JPanel implements ActionListener {
	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	DREM_Timeiohmm.Treenode rootptr;
	JRadioButton intersectButton, unionButton;
	int[] tfVals;
	DREMGui_PredictModel hmst;
	TableSorter sorter;
	DREM_Timeiohmm theTimeiohmm;
	DREMGui theDREMGui;
	int numrows, numcols;
	JButton hideButton;
	JCheckBox conditionalBox;
	JFrame theFrame;

	/**
	 * Constructs the prediction interface window
	 */
	public DREMGui_Predict(JFrame theFrame, DREMGui theDREMGui,
			final DREM_Timeiohmm.Treenode rootptr) {
		this.theFrame = theFrame;
		this.theDREMGui = theDREMGui;
		this.theTimeiohmm = theDREMGui.theTimeiohmm;
		this.rootptr = rootptr;
		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setBackground(bgColor);
		setForeground(fgColor);

		tfVals = theTimeiohmm.bindingData.signedBindingValuesSorted;
		numcols = tfVals.length + 1;
		numrows = theTimeiohmm.bindingData.regNames.length;

		Object[][] tabledata = new Object[numrows][numcols];
		String[] columnNames = new String[numcols];

		columnNames[0] = "TF";
		for (int ncol = 1; ncol < numcols; ncol++) {
			columnNames[ncol] = "" + tfVals[ncol - 1];
		}

		ButtonGroup[] group = new ButtonGroup[numrows];

		for (int nrow = 0; nrow < numrows; nrow++) {
			tabledata[nrow][0] = theTimeiohmm.bindingData.regNames[nrow];
			group[nrow] = new ButtonGroup();
			for (int ncol = 1; ncol < numcols; ncol++) {
				JRadioButton objButton = new JRadioButton("");
				objButton.setHorizontalAlignment(JLabel.CENTER);
				objButton.setBackground(Color.white);
				tabledata[nrow][ncol] = objButton;
				group[nrow].add(objButton);
			}
			((JRadioButton) tabledata[nrow][numcols / 2]).setSelected(true);
		}

		hmst = new DREMGui_PredictModel(tabledata, columnNames);
		sorter = new TableSorter(hmst);
		final JTable table = new JTable(sorter) {
			public void tableChanged(TableModelEvent e) {
				super.tableChanged(e);
				repaint();
			}
		};

		for (int ncol = 1; ncol < numcols; ncol++) {
			String szcolname = "" + tfVals[ncol - 1];
			table.getColumn(szcolname).setCellRenderer(
					new RadioButtonRenderer());
			table.getColumn(szcolname).setCellEditor(
					new RadioButtonEditor(new JCheckBox()));
		}

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
						* table.getRowCount(), 300)));
		// Add the scroll pane to this panel.
		add(scrollPane);

		JPanel conditionalPanel = new JPanel();
		conditionalBox = new JCheckBox(
				"Probabilities should be conditional on gene not being filtered");
		conditionalPanel.add(conditionalBox);
		conditionalBox.setBackground(Color.white);
		conditionalPanel.setBackground(Color.white);
		conditionalBox.setSelected(true);

		conditionalPanel.add(conditionalBox);
		add(conditionalPanel);
		conditionalPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));
		JPanel buttonPanel = new JPanel();
		buttonPanel.setBackground(Color.white);

		JButton showButton = new JButton("Show Prediction");
		showButton.setActionCommand("show");
		showButton.addActionListener(this);
		buttonPanel.add(showButton);

		hideButton = new JButton("Hide Prediction");
		hideButton.setActionCommand("hide");
		hideButton.addActionListener(this);
		hideButton.setEnabled(false);
		buttonPanel.add(hideButton);

		JButton defaultButton = new JButton("Default Settings");
		defaultButton.setActionCommand("default");
		defaultButton.addActionListener(this);
		buttonPanel.add(defaultButton);
		buttonPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 20));

		JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
		helpButton.addActionListener(this);
		helpButton.setActionCommand("help");
		buttonPanel.add(helpButton);

		add(buttonPanel);
	}

	/**
	 * Renderer for radio predict buttons
	 */
	static class RadioButtonRenderer implements TableCellRenderer {
		public Component getTableCellRendererComponent(JTable table,
				Object value, boolean isSelected, boolean hasFocus, int row,
				int column) {
			if (value == null)
				return null;
			return (Component) value;
		}
	}

	/**
	 * Editor for radio predict buttons
	 */
	static class RadioButtonEditor extends DefaultCellEditor implements
			ItemListener {
		private JRadioButton button;

		public RadioButtonEditor(JCheckBox checkBox) {
			super(checkBox);
		}

		public Component getTableCellEditorComponent(JTable table,
				Object value, boolean isSelected, int row, int column) {
			if (value == null)
				return null;
			button = (JRadioButton) value;
			button.addItemListener(this);
			return (Component) value;
		}

		public Object getCellEditorValue() {
			button.removeItemListener(this);
			return button;
		}

		public void itemStateChanged(ItemEvent e) {
			super.fireEditingStopped();
		}
	}

	/**
	 * Sets the value of all entries to bval
	 */
	void setAll(boolean bval) {
		for (int nrow = 0; nrow < numrows; nrow++) {
			for (int ncol = 1; ncol < numcols; ncol++) {
				hmst.setValueAt(Boolean.valueOf(bval), nrow, ncol);
			}
		}
	}

	/**
	 * Sets whether the probability on each hidden state is visible or not based
	 * on bvisible
	 */
	void setPredictTextVisible(DREM_Timeiohmm.Treenode treeptr, boolean bvisible) {
		if (treeptr != null) {
			treeptr.thepredictText.setVisible(bvisible);

			for (int nchild = 0; nchild < treeptr.numchildren; nchild++) {
				setPredictTextVisible(treeptr.nextptr[nchild], bvisible);
			}
		}
	}

	/**
	 * Responds to actions on the window
	 */
	public void actionPerformed(ActionEvent e) {

		String szcommand = e.getActionCommand();
		int[] predictinputs = new int[numrows];
		if (szcommand.equals("show")) {
			// build hash set of selected
			theDREMGui.bshowpredict = true;
			for (int nrow = 0; nrow < numrows; nrow++) {
				for (int ncol = 1; ncol < numcols; ncol++) {
					if (((JRadioButton) hmst.getValueAt(nrow, ncol))
							.isSelected()) {
						predictinputs[nrow] = tfVals[ncol - 1];
						break;
					}
				}
			}

			hideButton.setEnabled(true);
			try {
				double dprob;
				if (conditionalBox.isSelected()) {
					dprob = 1;
				} else {
					double[] dx;
					dx = theTimeiohmm.filteredClassifier
							.distributionForInstance(predictinputs);
					dprob = dx[1];
				}

				theTimeiohmm.predictTime(predictinputs, dprob, rootptr);
				setPredictTextVisible(rootptr, true);
			} catch (Exception ex) {
				ex.printStackTrace(System.out);
			}
		} else if (szcommand.equals("default")) {
			int nmidcol = numcols / 2;
			for (int nrow = 0; nrow < numrows; nrow++) {
				((JRadioButton) hmst.getValueAt(nrow, nmidcol))
						.setSelected(true);
			}
			repaint();
		} else if (szcommand.equals("hide")) {
			theDREMGui.bshowpredict = false;
			setPredictTextVisible(rootptr, false);
			hideButton.setEnabled(false);
		} else if (szcommand.equals("help")) {
			String szMessage = "DREM allows one to view for any set of transcription factor-gene regulation interaction inputs, "
					+ "the probability under the model of being in each state.  Through this dialog box one can specify a "
					+ "set of TF-gene interaction inputs for which they want to see the transition probabilities.  "
					+ "After pressing the button Show Prediction, the probabilities appear on the main interface.  "
					+ "The predictions then appear in the node of the states. Pressing the Hide Prediction button hides "
					+ "the predictions labels. Pressing the Default Settings button sets "
					+ "all input value for each transcription to `0'.  "
					+ "If the options Probabilities should be conditional on gene not being "
					+ "filtered, then the probabilities are computed conditional on the gene not being filtered. "
					+ "If the box is unchecked, then all probabilities are multiplied against the probability "
					+ "of a gene with the selected inputs not being filtered.  "
					+ "This probability of a gene not being filtered for a given set of inputs is "
					+ "determined using a Naive Bayes classifier.";

			Util.renderDialog(theFrame, szMessage, -200, -100);
		}
	}

}