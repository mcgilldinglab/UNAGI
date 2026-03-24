package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.core.*;
import javax.swing.*;
import javax.swing.event.*;
import java.awt.event.*;
import java.awt.*;

/**
 * Class controls the panel for displaying interface options
 */
public class GenePlotPanel extends JPanel implements ItemListener, ActionListener, ChangeListener
{

    ButtonGroup xgroup = new ButtonGroup();
    ButtonGroup ygroup = new ButtonGroup();
    ButtonGroup genescalegroup = new ButtonGroup();
    ButtonGroup genedisplaygroup = new ButtonGroup();
    JCheckBox scalevisibleBox;
    JCheckBox ymaintickBox;
    JCheckBox mainprofilelineBox;
    JCheckBox mainprofileIDBox;
    JCheckBox mainprofileDetailBox;
    JButton colorButton;
    JRadioButton uniformButton;
    JRadioButton realButton;
    JRadioButton automaticButton;
    JRadioButton fixedButton;
    JRadioButton geneButton;
    JRadioButton profileButton;
    JRadioButton globalButton;
    JRadioButton nodisplayButton;
    JRadioButton alwaysdisplayButton;
    JRadioButton onlyselectedButton;
    JLabel minLabel;
    JLabel maxLabel;
    JLabel tickLabel;
    JLabel tickmainLabel;
    JButton helpPolicyButton;
    JButton helpScaleButton;
    JButton helpXButton;
    JButton helpYWindowButton; 
    JSpinner  thespinnerminy; 
    JSpinner thespinnermaxy;
    JSpinner thespinnertick;
    JSpinner thespinnertickmain;
    boolean bkmeans;
    MAINGUI2 themaingui;
    JFrame thegeneplotFrame;

    /**
     * Constructor for the interface options
     */
    public GenePlotPanel(JFrame thegeneplotFrame,MAINGUI2 themaingui,boolean bkmeans)
    {
	this.thegeneplotFrame = thegeneplotFrame;
	this.themaingui = themaingui;
	this.bkmeans = bkmeans;

        setLayout(new BoxLayout(this, BoxLayout.PAGE_AXIS));
        setBackground(Color.white);

	JPanel genedisplayPanel = new JPanel();
	JLabel genedisplayLabel = new JLabel("           Display policy on main interface:            ");
	genedisplayPanel.setBackground(ST.lightBlue);
	genedisplayPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
	genedisplayPanel.add(genedisplayLabel);
	add(genedisplayPanel);


        genescalegroup = new ButtonGroup();
	nodisplayButton = new JRadioButton("Do not display genes");
        alwaysdisplayButton = new JRadioButton("Display all genes");
	mainprofileDetailBox = new JCheckBox("Display details when ordering",themaingui.bdisplaydetail);
	onlyselectedButton = new JRadioButton("Display only selected genes");
	JPanel maintickintervalpanel = new JPanel();
	ymaintickBox = new JCheckBox("Show Main Y-axis gene tick marks",themaingui.bdisplaymaintick);
        tickmainLabel = new JLabel("Main Y-axis gene tick interval");
        SpinnerNumberModel sntickmain = new SpinnerNumberModel(new Double(ST.dtickmainDEF),
                                                new Double(0.0000001),null,new Double(.5));

        helpPolicyButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpPolicyButton.addActionListener(this);
        helpPolicyButton.setActionCommand("help");

        thespinnertickmain = new JSpinner(sntickmain);
	maintickintervalpanel.add(tickmainLabel);
	maintickintervalpanel.add(thespinnertickmain);

	tickmainLabel.setBackground(Color.white);
        thespinnertickmain.setPreferredSize(new Dimension(60,24));
        thespinnertickmain.setMaximumSize(new Dimension(60,24));
	thespinnertickmain.setAlignmentX(Component.LEFT_ALIGNMENT);
	maintickintervalpanel.setAlignmentX(Component.LEFT_ALIGNMENT);
	thespinnertickmain.addChangeListener(this);

	if (bkmeans)
	{
           profileButton = new JRadioButton("Cluster specific");
	   mainprofilelineBox = new JCheckBox("Display Cluster Mean",themaingui.bdisplayprofileline);
	   mainprofileIDBox = new JCheckBox("Display K-Means Cluster ID",themaingui.bdisplayID);
	}
	else
	{
  	   geneButton = new JRadioButton("Gene specific");
           profileButton = new JRadioButton("Profile specific");
	   mainprofileIDBox = new JCheckBox("Display Profile ID",themaingui.bdisplayprofileline);
	   mainprofilelineBox = new JCheckBox("Display Model Profile",themaingui.bdisplayID);
	}

        globalButton = new JRadioButton("Global");
	scalevisibleBox = 
               new JCheckBox("Scale should be based on only selected genes");

	genedisplaygroup.add(nodisplayButton);
	genedisplaygroup.add(alwaysdisplayButton);
	genedisplaygroup.add(onlyselectedButton);
        colorButton = new JButton("Change Color of Genes");

	if (themaingui.ngenedisplay == 0)
	{
	   nodisplayButton.setSelected(true);
	   colorButton.setEnabled(false);
	   scalevisibleBox.setEnabled(false);

	   profileButton.setEnabled(false);
	   if (!bkmeans)
	   {
	      geneButton.setEnabled(false);
	   }
	   ymaintickBox.setEnabled(bkmeans);
	   globalButton.setEnabled(false);
	   thespinnertickmain.setEnabled(false);
	}
	else if (themaingui.ngenedisplay == 1)
	{
	   onlyselectedButton.setSelected(true);
	   colorButton.setEnabled(true);
	   boolean bscaleprofilesenabled = (themaingui.ngenescale > 0);
	   scalevisibleBox.setEnabled(bscaleprofilesenabled);

	   thespinnertickmain.setEnabled(bscaleprofilesenabled&&ymaintickBox.isSelected());
	   profileButton.setEnabled(true);
	   if (!bkmeans)
	   {
	      geneButton.setEnabled(true);
	      ymaintickBox.setEnabled(bscaleprofilesenabled);
	   }
	   else
	   {
	       ymaintickBox.setEnabled(true);
	   }
	   globalButton.setEnabled(true);
	}
	else
	{
	   boolean bymaintickenabled = (themaingui.ngenescale > 0)||(bkmeans);
	   ymaintickBox.setEnabled(bymaintickenabled);
	   thespinnertickmain.setEnabled(bymaintickenabled&&ymaintickBox.isSelected());
	   alwaysdisplayButton.setSelected(true);
	   colorButton.setEnabled(true);
	   scalevisibleBox.setEnabled(false);
	   profileButton.setEnabled(true);
	   if (!bkmeans)
	   {
	      geneButton.setEnabled(true);
	   }
	   globalButton.setEnabled(true);
	}

	tickmainLabel.setEnabled(thespinnertickmain.isEnabled());
	nodisplayButton.setBackground(Color.white);
	alwaysdisplayButton.setBackground(Color.white);
	onlyselectedButton.setBackground(Color.white);
        nodisplayButton.addItemListener(this);
        alwaysdisplayButton.addItemListener(this);
        onlyselectedButton.addItemListener(this);
	add(nodisplayButton);
	add(onlyselectedButton);
	add(alwaysdisplayButton);

	JPanel colorPanel = new JPanel();

        colorButton.setActionCommand("color");
        colorButton.setMinimumSize(new Dimension(800,20));
        colorButton.addActionListener(this);
	colorButton.setForeground(themaingui.thegenecolor);
	colorPanel.add(colorButton);
        colorPanel.add(helpPolicyButton);
	colorPanel.setBackground(Color.white);
	colorPanel.setAlignmentX(Component.LEFT_ALIGNMENT);

	add(colorPanel);
	mainprofileIDBox.setBackground(Color.white);
	mainprofilelineBox.setBackground(Color.white);
	mainprofileDetailBox.setBackground(Color.white);
	ymaintickBox.setBackground(Color.white);
	maintickintervalpanel.setBackground(Color.white);
	mainprofileIDBox.addActionListener(this);
	mainprofilelineBox.addActionListener(this);
	mainprofileDetailBox.addActionListener(this);
	ymaintickBox.addActionListener(this);

	add(mainprofilelineBox);
	add(mainprofileIDBox);
	add(mainprofileDetailBox);

	JPanel genescalePanel = new JPanel();
	JLabel genescaleLabel;

	if (bkmeans)
	{
           genescaleLabel = 
                  new JLabel("  Y-axis scale for genes on K-means main interface should be:   ");
	}
	else
        {
           genescaleLabel = 
                  new JLabel("  Y-axis scale for genes on main interface should be:  ");
	}

	genescalePanel.setBackground(ST.lightBlue);
	genescalePanel.setAlignmentX(Component.LEFT_ALIGNMENT);
	genescalePanel.add(genescaleLabel);
	add(genescalePanel);
        genescalegroup = new ButtonGroup();

        genescalegroup.add(globalButton);
	genescalegroup.add(profileButton);

        if (!bkmeans)
	{
 	   genescalegroup.add(geneButton);
	}


	if (themaingui.ngenescale == 0)
	{
	   if (bkmeans)
           {
              profileButton.setSelected(true);
	   }
	   else
	   {
	      geneButton.setSelected(true);
	   }
	}
	else if (themaingui.ngenescale == 1)
	{
	   profileButton.setSelected(true);
	}
	else
	{
	   globalButton.setSelected(true);
	}


	profileButton.setBackground(Color.white);
	globalButton.setBackground(Color.white);
        globalButton.addItemListener(this);
        profileButton.addItemListener(this);

	JPanel scalePanel = new JPanel();
	if (!bkmeans)
	{
	   geneButton.setBackground(Color.white);
           geneButton.addItemListener(this);
	   scalePanel.add(geneButton);
	}
	scalePanel.setBackground(Color.white);

	scalePanel.add(profileButton);
	scalePanel.add(globalButton);

	add(scalePanel);
	scalePanel.setAlignmentX(Component.LEFT_ALIGNMENT);

	scalevisibleBox.setBackground(Color.white);
	scalevisibleBox.setSelected(themaingui.bscalevisible);
	scalevisibleBox.addActionListener(this);
	add(scalevisibleBox);

        helpScaleButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpScaleButton.addActionListener(this);
        helpScaleButton.setActionCommand("help");
	maintickintervalpanel.add(helpScaleButton);

	add(ymaintickBox);
	add(maintickintervalpanel);

	JPanel XscalePanel = new JPanel();
	JLabel XscaleLabel = new JLabel("           X-axis scale should be:            ");
	XscalePanel.setBackground(ST.lightBlue);
	XscalePanel.setAlignmentX(Component.LEFT_ALIGNMENT);
	XscalePanel.add(XscaleLabel);

        xgroup = new ButtonGroup();
	JPanel ruPanel = new JPanel();
	realButton = new JRadioButton("Based on real time");
        uniformButton = new JRadioButton("Uniform");
	ruPanel.add(uniformButton);
	ruPanel.add(realButton);

	if (themaingui.binvalidreal)
	{
	    realButton.setEnabled(false);
	}

        helpXButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpXButton.addActionListener(this);
        helpXButton.setActionCommand("help");
        ruPanel.add(helpXButton);


	ruPanel.setBackground(Color.white);
	ruPanel.setAlignmentX(Component.LEFT_ALIGNMENT);
	xgroup.add(realButton);
	xgroup.add(uniformButton);
	if (themaingui.buniformXaxis)
	{
	   uniformButton.setSelected(true);
	}
	else
	{
	   realButton.setSelected(true);
	}
	uniformButton.setBackground(Color.white);
	realButton.setBackground(Color.white);
        uniformButton.addItemListener(this);
	realButton.addItemListener(this);

	JPanel YscalePanel = new JPanel();
	JLabel YscaleLabel;

	if (bkmeans)
	{
           YscaleLabel = new JLabel("           Y-axis scale on cluster details windows should be:            ");
	}
	else
	{
           YscaleLabel = new JLabel("           Y-axis scale on profile details windows should be:            ");
	}
	YscalePanel.setBackground(ST.lightBlue);
	YscalePanel.setAlignmentX(Component.LEFT_ALIGNMENT);
	YscalePanel.add(YscaleLabel);
	add(YscalePanel);

        ygroup = new ButtonGroup();
	automaticButton = new JRadioButton(" Determined automatically");
        fixedButton = new JRadioButton("Fixed with parameters below");
	ygroup.add(automaticButton);
	ygroup.add(fixedButton);

        SpinnerNumberModel snmaxy = new SpinnerNumberModel(new Double(ST.dyscalemaxDEF),new Double(0),
							   null,new Double(1));
        thespinnermaxy = new JSpinner(snmaxy);
        thespinnermaxy.setPreferredSize(new Dimension(60,24));
        thespinnermaxy.setMaximumSize(new Dimension(60,24));


        SpinnerNumberModel snminy = new SpinnerNumberModel(new Double(ST.dyscaleminDEF),null,new Double(0),
                                                                        new Double(1));
        thespinnerminy = new JSpinner(snminy);
        thespinnerminy.setPreferredSize(new Dimension(60,24));
        thespinnerminy.setMaximumSize(new Dimension(60,24));

        SpinnerNumberModel sntick = new SpinnerNumberModel(new Double(ST.dtickDEF),new Double(0.0000001),null,
                                                                        new Double(.5));
        thespinnertick = new JSpinner(sntick);
        thespinnertick.setPreferredSize(new Dimension(60,24));
        thespinnertick.setMaximumSize(new Dimension(60,24));

	thespinnermaxy.addChangeListener(this);
	thespinnerminy.addChangeListener(this);
	thespinnertick.addChangeListener(this);

        minLabel = new JLabel("Min:");
        maxLabel = new JLabel("Max:");
	tickLabel = new JLabel("Tick interval:");

	if (themaingui.bautomaticYaxis)
	{
	   automaticButton.setSelected(true);
	   thespinnermaxy.setEnabled(false);
	   thespinnerminy.setEnabled(false);
	   thespinnertick.setEnabled(false);

	   minLabel.setEnabled(false);
	   maxLabel.setEnabled(false);
	   tickLabel.setEnabled(false);
	}
	else
	{
	   fixedButton.setSelected(true);
	   thespinnermaxy.setEnabled(true);
	   thespinnerminy.setEnabled(true);
	   thespinnertick.setEnabled(true);

	   minLabel.setEnabled(true);
	   maxLabel.setEnabled(true);
	   tickLabel.setEnabled(true);
	}
	automaticButton.setBackground(Color.white);
	fixedButton.setBackground(Color.white);
        automaticButton.addItemListener(this);
	fixedButton.addItemListener(this);
	JPanel pautofix = new JPanel();

	pautofix.add(automaticButton);
	pautofix.add(fixedButton);
	pautofix.setAlignmentX(Component.LEFT_ALIGNMENT);
	pautofix.setBackground(Color.white);


	add(pautofix);
        JPanel p = new JPanel();
	p.add(minLabel);
        p.add(thespinnerminy);
	p.add(maxLabel);
        p.add(thespinnermaxy);
	p.add(tickLabel);
        p.add(thespinnertick);

        helpYWindowButton = new JButton(Util.createImageIcon("Help16.gif"));
        helpYWindowButton.addActionListener(this);
        helpYWindowButton.setActionCommand("help");
        p.add(helpYWindowButton);

	minLabel.setBackground(Color.white);
	maxLabel.setBackground(Color.white);
	tickLabel.setBackground(Color.white);
	p.setBackground(Color.white);
	p.setAlignmentX(Component.LEFT_ALIGNMENT);
	add(p);
	add(XscalePanel);
	add(ruPanel);
    }

    /**
     * Responds to stateChanged events
     */
    public void stateChanged(ChangeEvent e) 
    {
       if (e.getSource() == thespinnertickmain)
       {
	  themaingui.dtickintervalmain = ((Double) thespinnertickmain.getValue()).doubleValue();
	  themaingui.drawmain();
       }
       else
       {
          for (int nindex = 0; nindex < themaingui.openProfiles.size(); nindex++)
          {
             ((ProfileGui) themaingui.openProfiles.get(nindex)).repaint();
          }
       }
    }

    /**
     * Responds to itemStateChanged events
     */
    public void itemStateChanged(ItemEvent e) 
    {	
	Object source = e.getSource();
	boolean bignore =((source != scalevisibleBox)&&(source !=ymaintickBox) && 
                          (e.getStateChange() != e.SELECTED));
	//ignores unselected events of radio boxes
   
	if (!bignore)
        {
	   if ((source == automaticButton)||(source == fixedButton))
	   {
	      if (automaticButton.isSelected())
	      {
	         thespinnermaxy.setEnabled(false);
	         thespinnerminy.setEnabled(false);
	         thespinnertick.setEnabled(false);

       	         minLabel.setEnabled(false);
	         maxLabel.setEnabled(false);
	         tickLabel.setEnabled(false);
	      }
	      else
	      {
	         thespinnermaxy.setEnabled(true);
	         thespinnerminy.setEnabled(true);
	         thespinnertick.setEnabled(true);

  	         minLabel.setEnabled(true);
	         maxLabel.setEnabled(true);
	         tickLabel.setEnabled(true);
	      }
              for (int nindex = 0; nindex < themaingui.openProfiles.size(); nindex++)
	      {
	         ((ProfileGui) themaingui.openProfiles.get(nindex)).repaint();
	      }
	   }
	   else
	   {
	      if (realButton.isSelected())
  	      {
                 themaingui.buniformXaxis = false;
	      }
	      else
	      { 
                 themaingui.buniformXaxis = true;
	      }

	      if ((!bkmeans)&&(geneButton.isSelected()))
	      {
                 themaingui.ngenescale = 0;
	      }
	      else if (profileButton.isSelected())
	      {
                 themaingui.ngenescale = 1;
	      }
	      else
	      {
                 themaingui.ngenescale = 2;
	      }


  	      if (nodisplayButton.isSelected())
	      {
                 themaingui.ngenedisplay = 0;
	         colorButton.setEnabled(false);
	         scalevisibleBox.setEnabled(false);
		 ymaintickBox.setEnabled(bkmeans);
		 thespinnertickmain.setEnabled(false);
	         if (!bkmeans)
	         {
	            geneButton.setEnabled(false);
	         }
	         profileButton.setEnabled(false);
	         globalButton.setEnabled(false);
	      }
	      else if (onlyselectedButton.isSelected())
	      {
                 themaingui.ngenedisplay = 1;
	         colorButton.setEnabled(true);
	         if (bkmeans)
	         {
	    	    scalevisibleBox.setEnabled(true);
		    ymaintickBox.setEnabled(true); 
	         }
	         else
	         {
		    boolean bnotgenebutton = !geneButton.isSelected();
	            scalevisibleBox.setEnabled(bnotgenebutton);
		    ymaintickBox.setEnabled(bnotgenebutton); 
		    thespinnertickmain.setEnabled(bnotgenebutton&&ymaintickBox.isSelected());
	            geneButton.setEnabled(true);
	         }
	    
	         profileButton.setEnabled(true);
	         globalButton.setEnabled(true);
	      }
	     else 
	     {
                themaingui.ngenedisplay = 2;
	        colorButton.setEnabled(true);
	        scalevisibleBox.setEnabled(false);
		
		boolean bymaintickenabled =(bkmeans||!geneButton.isSelected());
    
		ymaintickBox.setEnabled(bymaintickenabled);
		thespinnertickmain.setEnabled(bymaintickenabled&&ymaintickBox.isSelected());
	        if (!bkmeans)
	        {
	           geneButton.setEnabled(true);
	        }
	        profileButton.setEnabled(true);
	        globalButton.setEnabled(true);
	     }

	      tickmainLabel.setEnabled(thespinnertickmain.isEnabled());
	      themaingui.drawmain();
	   }
	}
    }


    /**
     * Responds to actionPerformed events
     */
    public void actionPerformed(ActionEvent e)
    {
	String szCommand = e.getActionCommand();

	if (szCommand.equals("help"))
	{
	   Object esource = e.getSource();
	   String szMessage;

	   if (esource == helpPolicyButton)
	   {
               if (bkmeans)
	       {
	          szMessage = 
                   "The options in this section control the display "+
                   "policy on the main interface and the color of the genes if displayed. "+
		   "If 'Do not display genes' is selected then individual gene expression profiles are not "+
                   "shown on the main interface only the cluster mean profile is.  If 'Display only selected genes' is "+
                   "the selected option then indivdual gene expression profiles are only displayed when ordering the "+
                   "clusters by a GO category or gene set.  In this case only genes which "+
                   "belong to the selected GO category or gene set by which the ordering is based is displayed on "+
		   "the main interface.  If 'Display all genes' is selected then all genes not filtered are displayed.  "+
                   "If 'Display only selected' or 'Display all genes' is selected then there is the "+
                   "option to change the color of the genes on the main interface by pressing the 'Change Color of Genes' "+
		   "button.  The color of the text of this button will be the same color as the genes.\n\n"+
		   "When the 'Display Cluster Mean' option is selected the cluster average of the cluster is displayed.\n"+
                   "When the 'Display K-Means Cluster ID' option is selected the ID of the cluster in the upper left-hand corner\n"+
                   "is displayed.\n"+
                   "When 'Display details when ordering' is selected details about the cluster in the context of the "+
		   "ordering of the cluster is displayed in the lower left and upper right corners.";
 
	       }
	       else
	       {
	          szMessage = 
                   "The options in this section control the display "+
                   "policy on the main interface and the color of the genes if displayed. "+
		   "If 'Do not display genes' is selected then individual gene expression profiles are not "+
                   "shown on the main interface only the model profile is.  If 'Display only selected genes' is "+
                   "the selected option then indivdual gene expression profiles are only displayed when ordering the "+
                   "profiles or cluster of profiles by a GO category or gene set.  In this case only genes which "+
                   "belong to the selected GO category or gene set by which the ordering is based is displayed on "+
		   "the main interface.  If 'Display all genes' is selected then all genes not filtered are displayed.  "+
                   "If 'Display only selected' or 'Display all genes' is selected then there is the "+
                   "option to change the color of the genes on the main interface by pressing the 'Change Color of Genes' "+
		   "button.  The color of the text of this button will be the same color as the genes.\n\n"+
		   "When the 'Display Model Profile' option is selected the model profile pattern is displayed.\n"+
                   "When the 'Display ID' option is selected the ID of the model profile in the upper left-hand corner\n"+
                   "is displayed.\n"+
                   "When 'Display details when ordering' is selected details about the profile/cluster in the context of the "+
		      "ordering of the profiles is displayed in the lower left and/or upper right corners.";
	       }
	   }
	   else if (esource == helpScaleButton)
	   {
	       if (bkmeans)
	       {
		   szMessage = 
                     "The options in this section determines the y-axis scale of the individual gene expression "+
		     "profiles displayed on the main interface.  "+
                     "If 'Cluster specific' is selected then the y-scale of all genes in a cluster box are on the "+
                     "same scale, but the y-scale in different cluster boxes will be different. If 'Global' is selected "+
		     "then all genes are plotted on the same y-scale on the main interface.  Note that if there is one outlier gene "+
                     "and 'Cluster specific' is selected then the "+
                     "other genes in the cluster of the outlier will look flat, and if 'Global' is selected all other "+
                     "genes will look flat.\n\n"+
                     "If the gene display policy "+
		     "is to 'Display only selected', and 'Cluster specific' or 'Global' is selected then there is the further "+
		     "option to re-adjust the y-scale based on only the currently visible genes by selecting "+
		     "'Scale should be based on only selected genes'.  If genes appear on the main interface, then "+
                     "the cluster mean profile is adjusted to be on the same scale as the genes.\n\n"+
                     "When the option 'Show Main Y-axis gene tick marks' is selected then tick marks are visible "+
	             "at the interval 'Main Y-axis gene tick interval' where the longer tick mark in the center of the "+
	             "box is for 0.";
	       }
	       else
	       {
		   szMessage = 
                      "The options in this section determines the y-axis scale of the individual gene expression "+
		      "profiles displayed on the main interface.  If 'Gene specific' is selected then each individual "+
                      "gene is scaled separately to be closely aligned with the model profile.  This "+
                      "is valid since the correlation coefficient is used to measure distance "+
                      "and is unaffected by scaling.  "+
                      "If 'Profile specific' is selected then the y-scale of all genes in a profile box are on the "+
                      "same scale, but the y-scale in different profile boxes will be different. If 'Global' is selected "+
		      "then all genes are plotted on the same y-scale on the main interface.  Note that if there is one outlier gene "+
                      "and 'Profile specific' is selected then the "+
                      "other genes in the profile of the outlier will look flat, and if 'Global' is selected all other "+
                      "genes will look flat.\n\n"+
                      "If the gene display policy "+
		      "is to 'Display only selected', and 'Profile specific' or 'Global' is selected then there is the further "+
		      "option to re-adjust the y-scale based on only the currently visible genes by selecting "+
		      "'Scale should be based on only selected genes'.  Note that the model profiles will generally be on different "+
		      "scales than the genes.\n\n"+
                      "If the genes are displayed in a 'Profile specific' or 'Global' manner and the "+
                      "'Show Main Y-axis gene tick marks' is selected then tick marks corresponding to the gene "+
                      "expression values are visible "+
		      "at the interval 'Main Y-axis gene tick interval' where the longer tick mark in the center of the "+
	              "box is for 0.";
	       }
	   }
	   else if (esource ==  helpXButton)
	   {
	       if (bkmeans)
	       {
	          szMessage = 
                       "If this option is set to 'Uniform' all time points are "+
                       "placed at uniformly spaced intervals on the x-axis on both the main interface and "+
		       "the cluster details windows.  If this option is set to 'Based on real time' "+
                       "time points are placed on the x-axis proportionally spaced according to "+
                       "the real time points given in the column "+
                       "headers.  The time points needs to be in the same units.  If STEM was unable "+
	               "to parse the time points, then only the 'Uniform' option is active."; 
	       }
	       else
	       {
	          szMessage =  
                       "If this option is set to 'Uniform' all time points are "+
                       "placed at uniformly spaced intervals on the x-axis on both the main interface and "+
	               "the profile details windows.  If this option is set to 'Based on real time' "+
                       "time points are placed on the x-axis proportionally spaced according to "+
                       "the real time points given in the column "+
                       "headers.  The time points needs to be in the same units.  If STEM was unable "+
		       "to parse the time points, then only the 'Uniform' option is active."; 
	       }

	   }
	   else
	   {
	       if (bkmeans)
	       {
	          szMessage = 
                      "These options determine the y-axis scale on the cluster detail windows "+
                      "which appear when clicking on a cluster box on the main interface.  If "+
                      "'Determined automatically' is selected then STEM automatically determines the y-scale "+
                      "based on the expression level of the genes in the cluster.  The y-scale may "+
		      "be different for each cluster window.  If the option 'Fixed with parameters below' is selected "+
                      "then the y-scale on the cluster windows will have a minimum and maximum "+
                      "determined by the values of the 'Min' and 'Max' "+
                      "parameters respectively.  Additionally if 'Fixed with parameters below' "+
                      "option is selected the desired tick mark interval "+
	              "can also be specified through the 'Tick interval' parameter."; 

               }
	       else
	       {
                   szMessage = 
                      "These options determine the y-axis scale on the profile detail windows "+
                      "which appear when clicking on a profile box on the main interface.  If "+
                      "'Determined automatically' is selected then STEM automatically determines the y-scale "+
                      "based on the expression level of the genes assigned to the profile.  The y-scale may "+
	              "be different for each profile window.  If the option 'Fixed with parameters below' is selected "+
                      "then the y-scale on the profile windows will have a minimum and maximum "+
                      "determined by the values of the 'Min' and 'Max' "+
                      "parameters respectively.  Additionally if 'Fixed with parameters below' "+
                      "option is selected the desired tick mark interval "+
	              "can also be specified through the 'Tick interval' parameter."; 
	       }
	   }

	   Util.renderDialog(thegeneplotFrame,szMessage,-90,-15);
	}
        else if (szCommand.equals("color"))
	{
           Color newColor = JColorChooser.showDialog(
                     this,
                     "Choose Color",
                     themaingui.thegenecolor);
	     if (newColor != null)
	     {
		 themaingui.thegenecolor = newColor;
	         colorButton.setForeground(newColor);
 	         themaingui.drawmain();
	     }
	}
	else
	{
 	   themaingui.bscalevisible =  scalevisibleBox.isSelected();
	   themaingui.bdisplayID = mainprofileIDBox.isSelected();
	   themaingui.bdisplayprofileline = mainprofilelineBox.isSelected();
	   themaingui.bdisplaymaintick = ymaintickBox.isSelected();
	   themaingui.bdisplaydetail = mainprofileDetailBox.isSelected();
           thespinnertickmain.setEnabled((themaingui.ngenedisplay >= 1)
					 &&(themaingui.ngenescale >= 1)&&themaingui.bdisplaymaintick);
	   tickmainLabel.setEnabled(thespinnertickmain.isEnabled());
	   themaingui.drawmain();
	}
     }

}
