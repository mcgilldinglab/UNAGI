package edu.cmu.cs.sb.drem;

import edu.cmu.cs.sb.core.*;
import edu.umd.cs.piccolo.nodes.PPath;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.*;
import java.awt.event.*;
import java.util.*;
import java.text.NumberFormat;
import java.io.*;
import javax.imageio.*;
import java.awt.image.*;
import edu.umd.cs.piccolo.nodes.PImage;
import edu.umd.cs.piccolo.PCanvas;

import org.apache.batik.svggen.SVGGraphics2D;
import org.apache.batik.dom.GenericDOMImplementation;

import org.w3c.dom.Document;
import org.w3c.dom.DOMImplementation;

/**
 * Class for window to specify to save the current DREM main interface as an
 * image
 */
public class DREMGui_SaveDREM extends JPanel {

	final static Color bgColor = Color.white;
	final static Color fgColor = Color.black;
	DREMGui theDREMGui;

	HashSet validExt = new HashSet();

	/**
	 * Class constructor builds the panel window
	 */
	public DREMGui_SaveDREM(DREMGui theDREMGui, final JFrame theFrame) {
		this.theDREMGui = theDREMGui;

		setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
		setForeground(fgColor);
		final JFileChooser theChooser = new JFileChooser();
		String[] sznames = ImageIO.getWriterFormatNames();
		DREMGui_ImageFilter filter = new DREMGui_ImageFilter();
		StringBuffer szdescbuf = new StringBuffer();
		StringBuffer szvalidbuf = new StringBuffer();
		boolean bfirst = true;
		for (int nindex = 0; nindex < sznames.length; nindex++) {
			String szlower = sznames[nindex].toLowerCase(Locale.ENGLISH);
			if (!validExt.contains(szlower)) {
				filter.addExtension(szlower);
				validExt.add(szlower);
				if (!bfirst) {
					szdescbuf.append(", ");
					szvalidbuf.append(", ");
				}
				szdescbuf.append("*." + szlower);
				szvalidbuf.append(szlower);
				bfirst = false;
			}
		}

		filter.addExtension("svg");
		validExt.add("svg");
		if (!bfirst) {
			szdescbuf.append(", ");
			szvalidbuf.append(", ");
		}
		szdescbuf.append("*." + "svg");
		szvalidbuf.append("svg");

		final String szvalidext = szvalidbuf.toString();

		filter.setDescription("Image Files (" + szdescbuf.toString() + ")");
		theChooser.setFileFilter(filter);

		add(theChooser);
		theChooser.setAcceptAllFileFilterUsed(false);
		theChooser.setDialogType(JFileChooser.SAVE_DIALOG);
		final DREMGui ftheDREMGui = theDREMGui;

		theChooser.addActionListener(new ActionListener() {
			public void actionPerformed(ActionEvent e) {

				// set label's icon to the current image
				String state = (String) e.getActionCommand();

				if (state.equals(JFileChooser.CANCEL_SELECTION)) {
					theFrame.setVisible(false);
					theFrame.dispose();
				} else if (state.equals(JFileChooser.APPROVE_SELECTION)) {
					boolean bok = false;
					File f = theChooser.getSelectedFile();
					try {
						PCanvas canvas = ftheDREMGui.getCanvas();
						int width = (int) canvas.getCamera().getWidth();
						int height = (int) canvas.getCamera().getHeight();

						Image theImage = canvas.getLayer().toImage(width,
								height, Color.white);

						BufferedImage bi = PImage.toBufferedImage(theImage,
								true);

						String szext = DREMGui_ImageFilter.getExtension(f);
						if (szext.equalsIgnoreCase("svg")) {
							// Get a DOMImplementation.
							DOMImplementation domImpl = GenericDOMImplementation
									.getDOMImplementation();
							// Create an instance of org.w3c.dom.Document.
							String svgNS = "http://www.w3.org/2000/svg";
							Document document = domImpl.createDocument(svgNS,
									"svg", null);
							// Create an instance of the SVG Generator.
							SVGGraphics2D svgGenerator = new SVGGraphics2D(
									document);
							// svgGenerator = canvas.getGraphics();
							// canvas.update(svgGenerator);
							svgGenerator.setColor(Color.white);

							// PFrame theFrame = new PFrame("frame", false,
							// PCanvas aCanvas)
							ftheDREMGui.canvas.paintComponent(svgGenerator);
							// Ask the test to render into the SVG Graphics2D
							// implementation.
							// TestSVGGen test = new TestSVGGen();
							// test.paint(svgGenerator);
							// Finally, stream out SVG to the standard output
							// using
							// UTF-8 encoding.
							boolean useCSS = true; // we want to use CSS style
													// attributes
							Writer out = new OutputStreamWriter(
									new FileOutputStream(f), "UTF-8");
							svgGenerator.stream(out, useCSS);
							bok = true;
						} else if (validExt.contains(szext)) {
							ImageIO.write(bi, szext, f);
							bok = true;
						} else {
							final String fszext = szext;
							javax.swing.SwingUtilities
									.invokeLater(new Runnable() {
										public void run() {
											JOptionPane
													.showMessageDialog(
															null,
															fszext
																	+ " is not a recognized image extension.  "
																	+ "Recognized extensions are "
																	+ szvalidext
																	+ ".",
															"Invalid Image Extension",
															JOptionPane.ERROR_MESSAGE);
										}
									});
						}
					} catch (final IOException fex) {
						javax.swing.SwingUtilities.invokeLater(new Runnable() {
							public void run() {
								JOptionPane.showMessageDialog(null, fex
										.getMessage(), "Exception thrown",
										JOptionPane.ERROR_MESSAGE);
							}
						});
						fex.printStackTrace(System.out);
					}
					if (bok) {
						theFrame.setVisible(false);
						theFrame.dispose();
					}
				}
			}
		});
	}
}