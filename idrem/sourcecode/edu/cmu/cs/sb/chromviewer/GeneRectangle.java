package edu.cmu.cs.sb.chromviewer;
import java.awt.Color;

import java.util.ListIterator;
import java.util.Vector;
import java.util.Enumeration;
import edu.cmu.cs.sb.core.*;
import edu.umd.cs.piccolo.*;
import edu.umd.cs.piccolo.event.*;
import edu.umd.cs.piccolo.nodes.*;
import edu.umd.cs.piccolo.util.*;
import edu.umd.cs.piccolox.*;

    
/**
 * Draws gene rectangles on the main chromosome interface
 */
public class GeneRectangle extends PPath 
{    
   private GenomeFileParser yp;

   protected boolean mHorizontalOrientation = true; 
   
   public Gene mGeneInfo = null;
   public String mGeneAlias = null;

   protected double mWidthScale = 0;
   protected boolean mLeftToRight = true;

   protected GeneMouseOverHandler mInputListener = null;

    /**
     * Rectangle constructor
     */
   public GeneRectangle(Gene gene, String alias, float height, 
               double widthScale, boolean leftToRight,GenomeFileParser yp) 
   {
      mGeneInfo = gene;	
      mGeneAlias = alias;

      this.yp= yp;
      mLeftToRight = leftToRight;
      int nendstart = mGeneInfo.end-mGeneInfo.start+1;
      setPathToRectangle(0, 0, nendstart, height);
	
      setWidthScale(widthScale);
   }    

    /**
     * Handles various mouse events on the interface
     */
   public class GeneMouseOverHandler extends PBasicInputEventHandler 
   {
      private boolean mMouseIn = false;

      private PPath mHoverNode = null;
       
      private Vector genesUnder = new Vector();
	
      public void mouseClicked(PInputEvent e) 
      {
         if (!mMouseIn)
	    return;
	    
	 GeneRectangle picked = (GeneRectangle)e.getPickedNode();

         // get the bounds of the genes under the pointer
	 int vc_start = Integer.MAX_VALUE;
	 int vc_end = 0;
	    
         Enumeration egr = genesUnder.elements();
	 while(egr.hasMoreElements()) 
         {
	    GeneRectangle gr =(GeneRectangle) egr.nextElement();
		
	    int ngenestart =gr.mGeneInfo.start; 
	    int ngeneend =gr.mGeneInfo.end;

	    if (ngenestart < vc_start)
	       vc_start = ngenestart;
	    if (ngeneend > vc_end)
	       vc_end = ngeneend;
	 }
	    
         // species used in the ensembl URL
	 String[] species = yp.getSpecies().split(" ");
         // invoke the Ensembl gene map browser
         String ensemblViewer = "http://www.ensembl.org/";
         ensemblViewer += species[0] + "_" + species[1];
         ensemblViewer += "/contigview?chr=";	    
         ensemblViewer += (yp.getChroms())[picked.mGeneInfo.chromosome].namechrom;
         ensemblViewer += "&vc_start=" + vc_start + "&vc_end=" + vc_end;
         BareBonesBrowserLaunch.openURL(ensemblViewer);
      }

	
      public void mouseEntered(PInputEvent e) 
      {	    
	 mMouseIn = true;
	    
         try 
         {		
	    GeneRectangle picked = (GeneRectangle) e.getPickedNode();
	    GeneRectangle parent = (GeneRectangle) picked.getParent();

	    PBounds pBounds = picked.getGlobalBounds();
		
	    PText text = new PText("");
	    text.setTextPaint(picked.getPaint());

	    // to keep the text pretty and properly spaced
	    text.setText("");
	    // keep track of whether anything has been written
	    // to the tex node yet
	    boolean textFlag = false;

	    // find all siblings of the picked node that reside within
	    // 1 px of its borders 
	    ListIterator li = parent.getChildrenIterator();
	    while (li.hasNext()) 
            {
	       GeneRectangle sibling = (GeneRectangle)li.next();
	       PBounds sBounds = sibling.getGlobalBounds();
		    
	       // if sibling on a different strand, we don't care about it
	       if (picked.mGeneInfo.strand != sibling.mGeneInfo.strand)
	          continue;

	       if (sBounds.getX() <= pBounds.getX() + pBounds.getWidth() + 1 &&
		   sBounds.getX() + sBounds.getWidth() >= pBounds.getX() - 1) 
               {			
	          // if something already written in the box
		  // put in a new line before appending to it
		  if (textFlag)
	             text.setText(text.getText() + "\n");
		  textFlag = true;

		  int nstartval = sibling.mGeneInfo.start;
		  int nendval = sibling.mGeneInfo.end;

		  text.setText(text.getText() + 
			       sibling.getName() + "\n" + 
			       nstartval + ".." + nendval);
			
		  // store the pointer to this sibling in case something needs to be done later
		  genesUnder.add(sibling);

	       }
	    }		
		
	    mHoverNode = PPath.createRectangle(0,0,(float)text.getWidth() + 4,
						   (float)text.getHeight() + 4);

	    // paint the borders of the box
	    mHoverNode.setStrokePaint(Color.GRAY);
	    // add text to the box
	    mHoverNode.addChild(text);		
	    // offset text from the borders of the box
	    text.setOffset(2, 2);

	    // add the hovernode to the parent
	    parent.addChild(mHoverNode);		
	    // make sure the floating box gets painted on top of all adjacent elements
	    parent.moveToFront();
		
	    // it will be positioned by mouseMoved() which is called
	    // upon entrance also

	 } 
         catch (ClassCastException x) 
         {
	 }
      }
	
      public void mouseMoved(PInputEvent e) 
      {
         if (mHoverNode == null)
	    return;

	 try 
         {
	    // mHoverNode is a child of the chromosome rectangle
	    // and is positioned relative to it
	    // to access the rectangle, get the parent of the picked node
	    GeneRectangle parent = (GeneRectangle)e.getPickedNode().getParent();
		
	    // put the hover node at the mouse pointer
	    mHoverNode.setOffset(e.getPositionRelativeTo(parent));
	    // offset the box slightly so it is not obstructed by the mouse pointer
	    mHoverNode.translate(mHoverNode.getX() + 10, mHoverNode.getY() + 10);
		
	    mHoverNode.repaint();

	 } 
         catch (ClassCastException x) 
         {
	 }
      }


      public void mouseExited(PInputEvent e) 
      {
         if (!mMouseIn)
	    return;
	 mMouseIn = false;

	 if (mHoverNode == null)
	    return;
	    
	 try 
         {
	    // mHoverNode is a child of the chromosome rectangle
	    // and is positioned relative to it
	    // to access the rectangle, get the parent of the picked node
	    GeneRectangle parent = (GeneRectangle)e.getPickedNode().getParent();

	    parent.removeChild(mHoverNode);
	    mHoverNode = null;
		
	    // empty the vector of genes in the vicinity
	    genesUnder.clear();
	 } 
         catch (ClassCastException x) 
         {
	 }
      }
   }

    /**
     * Removes any genes in the rectangle
     */
   public void clearGenes() 
   {
      removeAllChildren();
   }

    /**
     * Sets the rectangle width
     */
   public void setWidthScale(double widthScale) 
   {
      if (widthScale == mWidthScale)
         return;

      mWidthScale = widthScale;
      int nendstart = mGeneInfo.end-mGeneInfo.start;
      setWidth(nendstart * mWidthScale);		
   }

    /**
     * Returns the chromosome corresponding to the gene in the rectangle
     */
    public String getName() 
    {
       if (mGeneAlias != null)
          return mGeneAlias;
	
       return mGeneInfo.namechrom;
    }
}
