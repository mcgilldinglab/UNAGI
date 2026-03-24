package edu.cmu.cs.sb.chromviewer;

import edu.cmu.cs.sb.core.*;
import edu.umd.cs.piccolo.*;
import edu.umd.cs.piccolo.event.*;
import edu.umd.cs.piccolo.nodes.*;
import edu.umd.cs.piccolo.util.*;
import edu.umd.cs.piccolox.*;

import java.awt.Color;

import java.util.ListIterator;

/**
 * Class encapsulates a rectangle for a chromosome
 * inherits some of the rectangle features from GeneRectangle
 */
public class ChromRectangle extends GeneRectangle 
{    
   private boolean mChangeLayout = false;

   private GenomeFileParser gp;

    /**
     * Class constructor
     */
   public ChromRectangle(Gene gene, float height, double widthScale, boolean leftToRight,GenomeFileParser gp) 
   {
      super(gene,null, height, widthScale, leftToRight,gp);
      this.gp = gp;
   }

    /**
     * Layouts out the genes on the rectangle
     */
   public void layoutChildren() 
   {
      if (!mChangeLayout)
         return;
      mChangeLayout = false;

      int cc = getChildrenCount();
	
      for (int i = 0; i < cc; i++) 
      {
         PNode pn = getChild(i);
         if (!pn.getClass().getName().equals("GeneRectangle"))
	    continue;

	 GeneRectangle gr = (GeneRectangle)pn;
	    
         gr.setWidthScale(mWidthScale);
	    
	 if (mLeftToRight)
	 {
	    int ngrstart= gr.mGeneInfo.start;
	    gr.setOffset(ngrstart * mWidthScale, gr.getYOffset());
	 }
	 else
	 {
	    int ndiffend = mGeneInfo.end -gr.mGeneInfo.end;
	    gr.setOffset(ndiffend * mWidthScale, gr.getYOffset());
	 }	    
      }
   }

    /**
     * Sets the chromosome width
     */
   public void setWidthScale(double widthScale) 
   {
      super.setWidthScale(widthScale);
      mChangeLayout = true;
   }

    /**
     * Adds a gene to this chromosome
     */
   public void addGene(Gene gene, String alias, java.awt.Paint color) 
   {
       GeneRectangle gr = new GeneRectangle(gene,
					     alias,
					     (float)getHeight() / 2 - 1, 
					     mWidthScale, 
					     mLeftToRight,gp);
      gr.setStrokePaint(color);
      gr.setPaint(color);
      gr.addInputEventListener(new GeneMouseOverHandler());
	
      double startX = 0;
      double startY = 0;

      if (mLeftToRight) 
      {
         startX = (double)gene.start * mWidthScale;
      }
      else
      {
         double ddiff = mGeneInfo.end-gene.end;
         startX = ddiff* mWidthScale;

      }

      if (gene.strand == '+')
         startY = 0;
      else
         startY = gr.getHeight();
	
      addChild(gr);
      gr.setOffset(startX, startY);

      repaint();
   }
}
