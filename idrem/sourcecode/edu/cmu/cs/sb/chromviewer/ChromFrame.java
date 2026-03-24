package edu.cmu.cs.sb.chromviewer;

import edu.umd.cs.piccolo.*;
import edu.umd.cs.piccolo.event.*;
import edu.umd.cs.piccolo.nodes.*;
import edu.umd.cs.piccolo.util.*;
import edu.umd.cs.piccolox.*;
import edu.umd.cs.piccolox.swing.*;

import edu.cmu.cs.sb.core.*;
import java.util.Vector;
import java.util.*;

import java.awt.Button;
import java.awt.Color;
import java.awt.BasicStroke;
import java.awt.Dimension;
import java.awt.Container;

import java.awt.event.*;

import javax.swing.JPanel;
import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.ScrollPaneConstants;
import javax.swing.BoxLayout;
import javax.swing.Box;
import javax.swing.WindowConstants;
import javax.swing.JColorChooser;


/**
 * The class for the main interface window of the chromosome viewer
 */
public class ChromFrame extends JFrame 
    implements ScrollPaneConstants, ActionListener 
{			
   private final static int CHROM_RECT_GAP = 20;
   private final static int CHROM_RECT_WIDTH = 20;
   private final static int BOUND_GAPLEFT = 10;
   private final static int BOUND_GAPRIGHT = 20;

   private int frame_width = 800;
   private int frame_height = 600;

   private double scale = 0;

   private boolean inited = false;

   public GenomeFileParser genomeParser = null;
   
   private ChromRectangle[] chromRectangles;

   private PText[] chromNames;
   private PPath   midLine;

   private PCanvas pc = null;

   private Color buttonColor = new Color(255,246,143);
   private Color geneColor = Color.RED;
   private JButton colorButton;

   private HashSet hsrejected;
   private HashMap hmaccepted;
   private HashMap hmcolor;

   private String szGeneHeader;
   private Gene[] chroms;

   private boolean bsortsize;

   private int[] sortMapping;

   private JButton buttonsort;

    /**
     * Class constructor
     */
   public ChromFrame(String szGeneHeader,GenomeFileParser genomeParser, 
                     int nwindowcount) 
   {
      setVisible(false);
      setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
      this.genomeParser = genomeParser;
      if (!genomeParser.szchromval.equals(""))
      {
         setTitle(genomeParser.getSpecies()+" ("+nwindowcount+")");	
         setSize(frame_width, frame_height);

         bsortsize = true;
         sortMapping = genomeParser.sortMappingSize;
         this.szGeneHeader = szGeneHeader;
         this.hsrejected = new HashSet();
         this.hmaccepted = new HashMap();
         this.hmcolor = new HashMap();
         this.genomeParser = genomeParser;

         JPanel contentPane = new JPanel();
         contentPane.setBackground(Color.white);
         contentPane.setLayout(new BoxLayout(contentPane,BoxLayout.PAGE_AXIS));
         setContentPane(contentPane);
	
         pc = new PCanvas();

         PScrollPane pscroll = new PScrollPane(pc, ScrollPaneConstants.VERTICAL_SCROLLBAR_AS_NEEDED,
					      ScrollPaneConstants.HORIZONTAL_SCROLLBAR_NEVER);
         pscroll.setBorder(new javax.swing.border.EmptyBorder(0,0,0,0));
         contentPane.add(pscroll);	
	
         JPanel buttonPane = new JPanel();
         buttonPane.setLayout(new BoxLayout(buttonPane,BoxLayout.LINE_AXIS));
         buttonPane.setBackground(Color.white);
	
         colorButton = new JButton("Next Gene Color");
         colorButton.setBackground(buttonColor);
         colorButton.setForeground(geneColor);
         colorButton.setActionCommand("color");
         colorButton.addActionListener(this);
         buttonPane.add(colorButton);
         buttonPane.add(Box.createRigidArea(new Dimension(5, 0)));

         JButton b1 = new JButton("Clear Genes");
         b1.setBackground(buttonColor);
         b1.setActionCommand("clear");
         b1.addActionListener(this);
         buttonPane.add(b1);
         buttonPane.add(Box.createRigidArea(new Dimension(5, 0)));

         buttonsort = new JButton("Sort By ID");
         buttonsort.setBackground(buttonColor);
         buttonsort.setActionCommand("sort");
         buttonsort.addActionListener(this);
         buttonPane.add(buttonsort);
         buttonPane.add(Box.createRigidArea(new Dimension(5,0)));

         JButton b2 = new JButton("Chr. Enrichment");
         b2.setBackground(buttonColor);
         b2.setActionCommand("enrichment");
         b2.addActionListener(this);
         buttonPane.add(b2);
         buttonPane.add(Box.createRigidArea(new Dimension(5, 0)));

         JButton b4 = new JButton("Gene Table");
         b4.setBackground(buttonColor);
         b4.setActionCommand("chromtable");
         b4.addActionListener(this);
         buttonPane.add(b4);
         buttonPane.add(Box.createRigidArea(new Dimension(5, 0)));

         JButton b5 = new JButton("Unmatched Genes");
         b5.setBackground(buttonColor);
         b5.setActionCommand("missinggenes");
         b5.addActionListener(this);
         buttonPane.add(b5);
         buttonPane.add(Box.createRigidArea(new Dimension(5, 0)));

         JButton helpButton = new JButton(Util.createImageIcon("Help16.gif"));
         helpButton.addActionListener(this);
         helpButton.setActionCommand("help");
         buttonPane.add(helpButton);	
         contentPane.add(buttonPane);

         drawChromosomes();	
      }
   }


    /**
     * Returns the canvas
     */
   private PCanvas getCanvas() 
   {
      return pc;
   }
	        
    /**
     * Empty initialize method
     */
   public void initialize() 
   {	
   }

    /**
     * to process resize of the frame
     */
   protected void processComponentEvent(ComponentEvent e) 
   {
      if (!inited)
      {
         return;
      }

      if (e.getID() == e.COMPONENT_RESIZED) 
      {
         redraw();
      }
   }

    /**
     * to process a button press
     */
   public void actionPerformed(ActionEvent e) 
   {
      String szcommand = e.getActionCommand();

      if (szcommand.equals("help"))
      {
         String szMessage = "Each rectangular box in the above window corresponds to a chromosome, "+
                            "and the small lines in these boxes correspond to genes. "+
                            "The top half of the box corresponds to its positive strand and the bottom half its negative strand."+
                            "When a 'Chromosome View' button on a gene table is pressed all genes on the table are plotted "+
                            "based on their location on the chromosome. "+
                            "The window accumulates genes until the 'Clear Genes' button is pressed. Mousing over a gene '"+
                            "gives its ID. Clicking on a gene opens the Ensembl genome browser to that gene\n\n"+
                   	    "Along the bottom of the window there are several buttons that function as follows "+
                            "The 'Next Gene Color' gives the option to change the color of the next genes displayed. "+
                            "Note that this does not change the color of the currently displayed genes.\n"+
                            "'Clear Genes' clears the current genes displayed in the chromosome viewer.\n"+
                            "'Sort by ID'/'Sort by Size'. If 'Sort by ID' button is visible then "+
                            "the chromosomes are currently sorted in decreasing order of size and pressing "+
                     	    "the button sorts them by increasing ID. If 'Sort by Size' button is visible then "+
                            "the chromosomes are currently sorted in increasing ID and pressing the button "+
	                    "sorts them by decreasing size.\n"+
                            "The 'Chr. Enrichment' button displays a table of chromosome enrichments for genes currently "+
                            "displayed.\n"+
                            "'Gene Table' button displays a table of the genes currently displayed in the chromosome viewer "+
                            "and their location.\n"+
                            "'Unmatched Genes' displays genes that were attempted to be displayed, but could not be matched "+
                            "to any chromosome location based on the chromosome location data.";
         Util.renderDialog(this,szMessage);
      }
      else if (szcommand.equals("chromtable"))
      {
         javax.swing.SwingUtilities.invokeLater(new Runnable() 
         {
            public void run() 
	    {
               String szTitle ="Table of Gene Locations";
               JFrame frame = new JFrame(szTitle);
               frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	       frame.setLocation(25,100);
               ChromGeneAcceptTable newContentPane = 
                           new ChromGeneAcceptTable(frame,hmaccepted,chroms,szGeneHeader);
               newContentPane.setOpaque(true); //content panes must be opaque
               frame.setContentPane(newContentPane);
               frame.pack();
               frame.setVisible(true);
	    }
	  });
      }
      else if (szcommand.equals("missinggenes"))
      {
         javax.swing.SwingUtilities.invokeLater(new Runnable() 
         {
            public void run() 
            {
               String szTitle ="Unmatched Genes";
               JFrame frame = new JFrame(szTitle);
               frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	       frame.setLocation(25,100);
               ChromGeneRejectTable newContentPane = new ChromGeneRejectTable(frame,hsrejected,szGeneHeader);
               newContentPane.setOpaque(true); //content panes must be opaque
               frame.setContentPane(newContentPane);
               frame.pack();
               frame.setVisible(true);
	    }
	 });
      }
      else if (szcommand.equals("sort"))
      {
	  if (bsortsize)
	  {
	      buttonsort.setText("Sort by Size");
	      bsortsize = false;
	      sortMapping = genomeParser.sortMappingID;
	  }
	  else
	  {
	      buttonsort.setText("Sort by ID");
	      bsortsize = true;
	      sortMapping = genomeParser.sortMappingSize;
	  }
	  redraw();
      }
      else if (szcommand.equals("enrichment")) 
      {
         final ChromFrame thecf = this;
         javax.swing.SwingUtilities.invokeLater(new Runnable() 
         {
            public void run() 
            {
               String szTitle ="Chromosome Enrichment";
               JFrame frame = new JFrame(szTitle);
               frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	       frame.setLocation(25,100);
               ChromEnrichmentTable newContentPane = 
                    new ChromEnrichmentTable(frame,thecf.genomeParser,hmaccepted.size());
               newContentPane.setOpaque(true); //content panes must be opaque
               frame.setContentPane(newContentPane);
               frame.pack();
               frame.setVisible(true);
	    }
	 });
      } 
      else if (szcommand.equals("clear"))
      {
         clearGenes();
      } 
      else  if (szcommand.equals("color"))
      {
         Color newColor = JColorChooser.showDialog(this,"Choose Color",geneColor);
	 if (newColor != null)
         {
            geneColor = newColor;
            colorButton.setForeground(newColor);
	 }
      }
   }
    
    /**
     * Adjust interface scale
     */
   private void resetScale(double factor) 
   {	
      if (factor != 0) 
      {
         scale *= factor;
         return;
      }

      if (inited)
      {
         frame_width = getCanvas().getWidth();
      }

      scale = ((float)frame_width - (BOUND_GAPLEFT +BOUND_GAPRIGHT)) / genomeParser.getChromLenMax();
   }	
    
    /**
     * Resizes the chromosomes
     */
   private void resizeChroms() 
   {
      resetScale(0);
	
      for (int i = 0; i < chromRectangles.length; i++) 
      {
         chromRectangles[i].setWidthScale(scale);
         positionChromRectangle(chromRectangles[i]);
	    
         chromNames[i].setOffset(chromRectangles[i].getXOffset(),
				    chromRectangles[i].getYOffset() - chromNames[i].getHeight());
      }
	
      repaint();
   }   

    /**
     * Positions the chromosome
     */
   private void positionChromRectangle(ChromRectangle gr) 
   {
      int chromosome =sortMapping[gr.mGeneInfo.chromosome];

      float startX = 0;
      float startY = 0;
	
      startX = BOUND_GAPLEFT;
      startY = CHROM_RECT_WIDTH + (CHROM_RECT_WIDTH * 2 * chromosome);
	
      gr.setOffset(startX, startY);
   }
	
    /**
     * Draws the chromosomes on the interface window
     */
   private void drawChromosomes() 
   {
      chroms = genomeParser.getChroms();	

      chromRectangles = new ChromRectangle[chroms.length];
      chromNames = new PText[chroms.length];

      float startY = 0;
      float startX = 0;	
	
      PLayer layer = getCanvas().getLayer();

      resetScale(0);
      for (int i = 0; i < chroms.length; i++) 
      {
         chromRectangles[sortMapping[i]] = new ChromRectangle(chroms[i], 
						    CHROM_RECT_WIDTH, 
						    scale, 
						    true,genomeParser);
         layer.addChild(chromRectangles[sortMapping[i]]);
	    
	 chromNames[sortMapping[i]] = new PText(chroms[i].namechrom);
         layer.addChild(chromNames[sortMapping[i]]);
      }	
	
      resizeChroms();
      inited = true;
	
      // do it again on initialization
      resizeChroms();
   }
    
    /**
     * Draws geneNames on the interface window 
     */
   public void drawGenes(String[] geneNames) 
   {
      setVisible(true);
      drawChromosomes();
      drawGenes(geneNames, geneColor);
   }

    /**
     * Redraws the current genes and chromosomes on the interface
     */
   public void redraw()
   {
      PCanvas canvas = getCanvas();
      PLayer theLayer = canvas.getLayer();
      theLayer.removeAllChildren();
      drawChromosomes();
      Iterator hmaccepteditr = hmaccepted.keySet().iterator();
      while (hmaccepteditr.hasNext())
      {
         String szgenename = (String) hmaccepteditr.next();
	 HashSet thegeneset = (HashSet) hmaccepted.get(szgenename);
         Color thecolor = (Color) hmcolor.get(szgenename);

	 Iterator thegenesetitr = thegeneset.iterator();	

	 while (thegenesetitr.hasNext())
	 {
	    Gene thegene = (Gene) thegenesetitr.next();
	    chromRectangles[sortMapping[thegene.chromosome]].addGene(
					     thegene,szgenename,thecolor);
	 }
      }
   }

    /**
     * Draws geneNames on the interface with the color "color"
     */
   public void drawGenes(String[] geneNames, Color color) 
   {
      HashSet[] genes = genomeParser.addActiveGenes(geneNames);
	
      for (int i = 0; i < genes.length; i++) 
      {
         if (genes[i] == null)
	 {
	    hsrejected.add(geneNames[i]);
	 }
	 else if (hmaccepted.get(geneNames[i])==null)
	 {
	    hmaccepted.put(geneNames[i],genes[i]);
	    hmcolor.put(geneNames[i],color);

	    Iterator geneitr = (Iterator) genes[i].iterator();
	    while (geneitr.hasNext())
	    {
		Gene thegene = (Gene) geneitr.next(); 
		chromRectangles[sortMapping[thegene.chromosome]].addGene(
						       thegene,geneNames[i],color);
	    }
	 }
      }

      repaint();      
   }
	
    /**
     * Removes genes from the interface
     */ 
   public void clearGenes() 
   {	
      hsrejected = new HashSet();
      hmaccepted = new HashMap();
      hmcolor = new HashMap();
      for (int i = 0; i < chromRectangles.length; i++)
      {
         chromRectangles[i].clearGenes();
      }
      genomeParser.clearActiveGenes();
      redraw();	
   }
}
