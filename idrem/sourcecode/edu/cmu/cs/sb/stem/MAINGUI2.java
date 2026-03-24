package edu.cmu.cs.sb.stem;

import java.awt.Color;
import edu.cmu.cs.sb.core.*;
import edu.cmu.cs.sb.chromviewer.*;
import edu.umd.cs.piccolo.PCanvas;
import edu.umd.cs.piccolo.PNode;
import edu.umd.cs.piccolo.event.PZoomEventHandler;
import edu.umd.cs.piccolo.event.PBasicInputEventHandler;
import edu.umd.cs.piccolo.event.PInputEvent;
import edu.umd.cs.piccolo.nodes.PPath;
import edu.umd.cs.piccolox.PFrame;
import edu.umd.cs.piccolo.nodes.PText;
import edu.umd.cs.piccolo.nodes.PImage;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import javax.swing.*;
import java.util.*;
import java.text.NumberFormat;
import java.text.*;
import java.io.*;
import java.net.*;

import org.apache.batik.svggen.SVGGraphics2D;
import org.apache.batik.dom.GenericDOMImplementation;

import org.w3c.dom.Document;
import org.w3c.dom.DOMImplementation;

/**
 * The class for the overview interface window that shows all the profiles or k-means clusters
*/
public class MAINGUI2 extends PFrame implements ActionListener
{
    static int SCREENWIDTH = 800;
    static int SCREENHEIGHT = 600;
    static int BUFFERLEFT = 30;
    static int BUFFERRIGHT = 30; 
    static int BUFFERTOP = 25;
    static int BUFFERBOTTOM = 50;
 
    final static int maxCharHeight = 15;
    final static int minFontSize = 6;

    int[] profilelookup;
    int[] ranklookup;
    int[] cluster;  //cluster each profile belongs to 
    ProfileSortRec[] theProfileSortRec;
    int ngenescale=1; //0 gene; 1-profile; 2-scale 
    int ngenedisplay; //0-no display; 1-only if selected; 2-except if not selected; 3-always display; 
    boolean bscalevisible = ST.bscalevisibleDEF;//true;

    PCanvas canvas;

    Color thegenecolor = new Color(ST.genecolorRDEF,ST.genecolorGDEF,ST.genecolorBDEF); 

    JDialog theOptions;

    JFrame thegeneplotframe;
    JFrame profilesortframe;
    JFrame clustersortframe;
    GenePlotPanel thegeneplotpanel;

    CompareGui thecomparegui;

    PNode[] profilenodes;
    boolean bsaveprofile = true;
    Color[] savedcolors = null;
    ArrayList openProfiles = new ArrayList();

    boolean binvalidreal = false;
    ChromFrame cf;

    Color clustercolors[] = {
	new Color((float) 255/255,(float) 0/255,(float) 0/255,(float) .4),//red
	new Color((float) 0/255,(float) 255/255,(float) 0/255,(float) .4),//green
	new Color((float) 0/255,(float) 0/255,(float) 255/255,(float) .4),//blue
	new Color((float) 255/255,(float) 255/255,(float) 0/255,(float) .4),//yellow
	new Color((float) 255/255,(float) 128/255,(float) 0/255,(float) .4),//orange
	new Color((float) 255/255,(float) 0/255,(float) 255/255,(float) .4),//pink
	new Color((float) 128/255,(float) 0/255,(float) 0/255,(float) .4),//brown-red
	new Color((float) 0/255,(float) 0/255,(float) 128/255,(float) .4),//blue
	new Color((float) 0/255,(float) 128/255,(float) 0/255,(float) .4),//green
	new Color((float) 128/255,(float) 0/255,(float) 128/255,(float) .4),//purple  
	new Color((float) 128/255,(float) 128/255,(float) 128/255,(float) .4),//gray  
	new Color((float) 255/255,(float) 128/255,(float) 128/255,(float) .4),//pink
	new Color((float) 0/255,(float) 128/255,(float) 255/255,(float) .4),//darker blue
	new Color((float) 128/255,(float) 128/255,(float) 0/255,(float) .4),//beige
	new Color((float) 128/255,(float) 255/255,(float) 128/255,(float) .4),//light green
	new Color((float) 128/255,(float) 0/255,(float) 255/255,(float) .4),//purple
	new Color((float) 0/255,(float) 255/255,(float) 128/255,(float) .4),//blue-green
	new Color((float) 128/255,(float) 128/255,(float) 255/255,(float) .4),//blue
	new Color((float) 128/255,(float) 255/255,(float) 0/255,(float) .4),//green
	new Color((float) 0/255,(float) 255/255,(float) 255/255,(float) .4),//light blue
        new Color((float) 255/255,(float) 128/255,(float) 255/255,(float) .4),
        new Color((float) 0/255,(float) 128/255,(float) 128/255,(float) .4),
        new Color((float) 128/255,(float) 255/255,(float) 255/255,(float) .4),
        new Color((float) 255/255,(float) 0/255,(float) 128/255,(float) .4),
        new Color((float) 255/255,(float) 255/255,(float) 128/255,(float) .4)};

    static ArrayList coloroverflow = new ArrayList();
    static int NUSED = 0;
    STEM_DataSet theDataSet = null;
    static Object datasetlock = new Object();
    static int nwindowcount = 1;
    static int nopen = 0;

    boolean buniformXaxis = ST.buniformXaxisDEF;
    boolean bautomaticYaxis = ST.bautomaticYDEF;

    boolean bdisplayID = ST.bdisplayIDDEF;
    boolean bdisplayprofileline = ST.bdisplayprofilelineDEF;
    boolean bdisplaymaintick = ST.bdisplaymaintickDEF;
    boolean bdisplaydetail = ST.bdisplaydetailDEF;
    double dtickintervalmain = ST.dtickmainDEF;

    SpinnerNumberModel snmsigLevel = new SpinnerNumberModel(new Double(ST.dCompareMinpvalDEF),new Double(0), 
                                                      new Double(1), new Double(0.001));
    SpinnerNumberModel snmnumgenes  = new SpinnerNumberModel(new Integer(ST.nCompareMinGenesDEF),new Integer(1), 
                                                      null, new Integer(1));
    JSpinner thespinnersigLevel = new JSpinner(snmsigLevel);
    JSpinner thespinnernumgenes = new JSpinner(snmnumgenes);
    JButton browse1Button =  new JButton("Browse...", Util.createImageIcon("Open16.gif"));
    JButton compareButtonDialog = new JButton("Compare");
    JLabel compare1Label  = new JLabel("Comparison Data File:",JLabel.TRAILING);
    JLabel sigLevelLabel  = new JLabel("Maximum uncorrected intersection p-value:",JLabel.TRAILING);
    JLabel numgenesLabel  = new JLabel("Minimum number of genes in intersection:",JLabel.TRAILING);

    JTextField compare1Field = new JTextField(ST.szCompareDEF, JLabel.TRAILING);

    JButton executecompareHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton compare1HButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton compare2HButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton sigLevelHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton numgenesHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton viewHButton = new JButton(Util.createImageIcon("Help16.gif"));
    JButton viewButton = new JButton("View Comparison Data File"); 
    JButton repeatButton = new JButton("Comparison Repeat Data...", Util.createImageIcon("Open16.gif"));
    ListDialog theCompareRepeatList = new ListDialog(this,ST.vRepeatCompareDEF, 
			       ST.bcomparealltimeDEF,repeatButton,
			       ST.lightBlue,ST.defaultColor,ST.fc);

    /**
     * Class constructor
     */
    public MAINGUI2(STEM_DataSet theDataSet) 
    {
       super((theDataSet.bkmeans ? "All K-Means Clusters": "All STEM Profiles")+" ("+nwindowcount+")"); 

       synchronized(datasetlock)
       { 
          this.theDataSet = theDataSet;
	  ngenedisplay = ST.ngenedisplayDEF;
	  if (theDataSet.bkmeans)
	  {
	     ngenescale = ST.ngenescalekmeansDEF;
          }
	  else
	  {
             ngenescale = ST.ngenescaleDEF;
          }

          nwindowcount++;
	  nopen++;
          makeCompareDialog();

          cluster = new int[theDataSet.modelprofiles.length];
          theProfileSortRec = new ProfileSortRec[theDataSet.modelprofiles.length];
          profilelookup = new int[cluster.length];  //maps rank to profile id
          ranklookup    = new int[cluster.length];  //maps profile id to rank
 	  datasetlock.notifyAll();
       }

       addWindowListener(new WindowAdapter()
       {
          public void windowClosing(WindowEvent e) 
          {
	     synchronized(datasetlock)
	     { 
	        nopen--;
                if (nopen == 0)
	        {
	           nwindowcount = 1;
                   coloroverflow = new ArrayList();
		   NUSED =0;
		}
	     }
	  }
       });
    }

    /**
     * Calls the chromosome viewer constructor
     */
    public void chromViewInit(GenomeFileParser gfp) 
    {
       cf = new ChromFrame(theDataSet.szGeneHeader,gfp,nwindowcount-1);
    }
       
    /**
     * Record with information about a profile
     */
    static class ProfileSortRec
    {
       int ncluster;
       double dgenes;
       int nprofileindex;
       double dpval;
       double dexpectedgenes;

       ProfileSortRec(int ncluster, int nprofileindex, double dpval,double dgenes, double dexpectedgenes)
       {
          this.ncluster = ncluster;
	  this.nprofileindex = nprofileindex;
	  this.dpval = dpval;
          this.dgenes = dgenes;
          this.dexpectedgenes = dexpectedgenes;
       }
    }
        
    /**
     * The default profile comparison. Profiles are by default ordered based on 
     * first the cluster ID, then p-value, then number of genes assigned, then number expected
     * then profile ID
     */
    public static class ProfileComparator implements Comparator 
    {
       public int compare(Object o1, Object o2)
       {
          ProfileSortRec pso1 = (ProfileSortRec) o1;
          ProfileSortRec pso2 = (ProfileSortRec) o2;

          if (pso1.ncluster < pso2.ncluster)
             return -1;
          if (pso1.ncluster > pso2.ncluster)
             return 1;
          if (pso1.dpval < pso2.dpval)
             return -1;
          if (pso1.dpval > pso2.dpval)
             return 1;
          if (pso1.dgenes > pso2.dgenes)
             return -1;
          if (pso1.dgenes < pso2.dgenes)
             return 1;
          if (pso1.dexpectedgenes < pso2.dexpectedgenes)
             return -1;
          if (pso1.dexpectedgenes > pso2.dexpectedgenes)
             return 1;
          if (pso1.nprofileindex < pso2.nprofileindex)
             return -1;
          if (pso1.nprofileindex > pso2.nprofileindex)
             return 1;               
           return 0;
       }
    }

    /**
     * Compares profiles based on profile IDs
     */
    public static class ProfileIDComparator implements Comparator 
    {
       public int compare(Object o1, Object o2)
       {
          ProfileSortRec pso1 = (ProfileSortRec) o1;
          ProfileSortRec pso2 = (ProfileSortRec) o2;
  
          if (pso1.nprofileindex < pso2.nprofileindex)
	      return -1;
          if (pso1.nprofileindex > pso2.nprofileindex)
	      return 1;              
	  return 0;
       }
    }

    /**
     * Compares profiles based on number of genes assigned then by ID
     */
    public static class ProfilenumComparator implements Comparator 
    {
       public int compare(Object o1, Object o2)
       {
          ProfileSortRec pso1 = (ProfileSortRec) o1;
          ProfileSortRec pso2 = (ProfileSortRec) o2;

          if (pso1.dgenes < pso2.dgenes)
             return 1;
          if (pso1.dgenes > pso2.dgenes)
             return -1;  
          if (pso1.nprofileindex < pso2.nprofileindex)
             return -1;
          if (pso1.nprofileindex > pso2.nprofileindex)
             return 1;               
          return 0;
       }
    }

    /**
     * Compares profiles based on expectected number of genes assigned then by ID
     */
    public static class ProfileexpComparator implements Comparator 
    {
       public int compare(Object o1, Object o2)
       {
	   ProfileSortRec pso1 = (ProfileSortRec) o1;
	   ProfileSortRec pso2 = (ProfileSortRec) o2;

           if (pso1.dexpectedgenes < pso2.dexpectedgenes)
              return 1;
           if (pso1.dexpectedgenes > pso2.dexpectedgenes)
              return -1;  
           if (pso1.nprofileindex < pso2.nprofileindex)
              return -1;
           if (pso1.nprofileindex > pso2.nprofileindex)
              return 1;               
           return 0;
       }
    }

    /**
     * Compares profiles based on the p-valu signficance, then based on number of
     * genes assigned, number expected, and then ID
     */
    public static class ProfilesigComparator implements Comparator 
    {
       public int compare(Object o1, Object o2)
       {
          ProfileSortRec pso1 = (ProfileSortRec) o1;
          ProfileSortRec pso2 = (ProfileSortRec) o2;

          if (pso1.dpval < pso2.dpval)
             return -1;
          if (pso1.dpval > pso2.dpval)
             return 1;
          if (pso1.dgenes < pso2.dgenes)
             return 1;
          if (pso1.dgenes > pso2.dgenes)
             return -1;
          if (pso1.dexpectedgenes < pso2.dexpectedgenes)
             return -1;
          if (pso1.dexpectedgenes > pso2.dexpectedgenes)
             return 1;  
          if (pso1.nprofileindex < pso2.nprofileindex)
             return -1;
          if (pso1.nprofileindex > pso2.nprofileindex)
             return 1;
          return 0;
       }
    }
    
    /**
     * Sets the screenSize
     */
    public void beforeInitialize()
    {
       setSize(SCREENWIDTH,SCREENHEIGHT);
    }
           
    /**
     * Initializes the interface
     */
    public void initialize()
    {  
	drawmain();
    }

    /**
     * Closes the windows to sort profiles or change interface options
     */
    public void closeSortWindows()
    {
       if (profilesortframe != null)
       {
          cf.setVisible(false);
	  cf.dispose();
          cf = null;
       }

       if (profilesortframe != null)
       {
          profilesortframe.setVisible(false);
	  profilesortframe.dispose();
	  ((SortTable) profilesortframe.getContentPane()).closeDefineWindows();
          profilesortframe = null;
       } 

       if (clustersortframe != null)
       {
          clustersortframe.setVisible(false);
	  clustersortframe.dispose();
	  ((SortTable) clustersortframe.getContentPane()).closeDefineWindows();
          clustersortframe = null;
       }  

       if (thegeneplotframe != null)
       {
	   thegeneplotframe.setVisible(false);
	   thegeneplotframe.dispose();
	   thegeneplotframe = null;
       }

       //added to fix memory leak
       if (thegeneplotpanel != null)
       {
          thegeneplotpanel.themaingui = null;
	  thegeneplotpanel = null;
       }  

       if (cf != null)
       {
          cf.dispose();
      	  cf = null;
       }
    }

    /////////////////////////////////////////////////////////////////////////
    /**
     * Converts a double value into a string
     */
    public static String doubleToSz(double dval)
    {
       String szexp;
       double dtempval = dval;
       int nexp = 0;
       NumberFormat nf0 = NumberFormat.getInstance(Locale.ENGLISH);
       nf0.setMinimumFractionDigits(0);
       nf0.setMaximumFractionDigits(0);

       NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);
       nf1.setMinimumFractionDigits(1);
       nf1.setMaximumFractionDigits(1);

       if (dval <= 0)
       {
          szexp = "0.0";
       }
       else if (dval > 0.095)
       {
	  szexp = nf1.format(dval);
       }
       else 
       {
          while ((dtempval<0.95)&&(dtempval>0))
          {
             nexp--;
             dtempval = dtempval*10;
          }
          dtempval = Math.pow(10,Math.log(dval)/Math.log(10)-nexp);
          szexp = nf0.format(dtempval)+"e"+nexp;   
       }	
       return szexp;
    }

    ////////////////////////////////////////////////////////////////////////////////
    /**
     * Renders the main interface window
     */
    public void drawmain()
    {
       synchronized(datasetlock)
       {
          while (theDataSet == null)
          {
             try
	     {
	        datasetlock.wait();
             } 
             catch (InterruptedException e) 
             {
             }
	  }

	  theDataSet.tga.tpgr= null;

          int numclusters = theDataSet.clustersofprofilesnum.size();
          for (int nindex = 0; nindex < theDataSet.modelprofiles.length; nindex++)
	  {
             cluster[nindex] = numclusters;
             theProfileSortRec[nindex] = new ProfileSortRec(numclusters,nindex,
							     theDataSet.pvaluesassignments[nindex],
                                                             theDataSet.countassignments[nindex],
                                                             theDataSet.expectedassignments[nindex]);  
	  }     

          for (int ncluster= 0; ncluster< theDataSet.clustersofprofilesnum.size(); ncluster++)
          {
	     ArrayList profilesInCluster = (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
             for (int nprofileindex = 0; nprofileindex < profilesInCluster.size(); nprofileindex++)
	     {
	        STEM_DataSet.ProfileRec theProfile = (STEM_DataSet.ProfileRec) profilesInCluster.get(nprofileindex);
                cluster[theProfile.nprofileindex] = ncluster;
                theProfileSortRec[theProfile.nprofileindex] = 
                                   new ProfileSortRec(ncluster,theProfile.nprofileindex,
				    theDataSet.pvaluesassignments[theProfile.nprofileindex],
				    theDataSet.countassignments[theProfile.nprofileindex],
                                    theDataSet.expectedassignments[theProfile.nprofileindex]);               
	     }
	  }      
      
          //sort profiles by cluster number
          //want a lookup table for profile position to actual profile
          if (theDataSet.tga.szsortcommand.equals("exp"))
	  {
	     Arrays.sort(theProfileSortRec, new ProfileexpComparator());
	  }
          else if (theDataSet.tga.szsortcommand.equals("num"))
          {
             Arrays.sort(theProfileSortRec, new ProfilenumComparator());
          }
          else if (theDataSet.tga.szsortcommand.equals("sig"))
          {
             Arrays.sort(theProfileSortRec, new ProfilesigComparator());
          }
          else if  ((theDataSet.tga.szsortcommand.equals("id"))||
		      (theDataSet.tga.szsortcommand.equals("default"))&&(theDataSet.bkmeans))
          {
	     Arrays.sort(theProfileSortRec, new ProfileIDComparator());
	  }
          else if (theDataSet.tga.szsortcommand.equals("default"))
	  {
             Arrays.sort(theProfileSortRec, new ProfileComparator());
          }
          else
	  {
	     theDataSet.tga.getProfileRankings(theDataSet.clustersofprofilesnum,
                                                    theDataSet.profilesAssigned, theDataSet.genenames,
						    theDataSet.bestassignments,
                                                    theDataSet.expectedassignments);

             ProfileSortRec[] tempProfileSortRec = new ProfileSortRec[theProfileSortRec.length];   
             for (int nindex = 0; nindex < theProfileSortRec.length; nindex++)
	     {      
	        tempProfileSortRec[nindex] = 
                       theProfileSortRec[((GoAnnotations.ProfileGORankingRec)  theDataSet.tga.tpgr[nindex]).nprofile];
	     }

             theProfileSortRec = tempProfileSortRec;
	  }

          for (int nindex = 0; nindex < profilelookup.length; nindex++)
          {
             profilelookup[nindex] = theProfileSortRec[nindex].nprofileindex;
             ranklookup[theProfileSortRec[nindex].nprofileindex] = nindex;
	  }

          PNode node; 
          double dmainwidth = SCREENWIDTH - BUFFERLEFT - BUFFERRIGHT;
          double dmainheight = SCREENHEIGHT - BUFFERTOP - BUFFERBOTTOM;

          double dunitcount = theDataSet.numcols;

          double dterm1 = dunitcount*dmainheight+dmainwidth*dunitcount;
          double dterm2 = Math.sqrt(Math.pow(dunitcount*dmainheight+dmainwidth*dunitcount,2)
                     -4*dmainheight*dmainwidth*(Math.pow(dunitcount,2)-
                       theDataSet.modelprofiles.length*Math.pow(dunitcount,2)));
          double dterm3,ddeltax;
          if (theDataSet.modelprofiles.length > 1) 
	  {
             dterm3 = 2*(dunitcount*dunitcount*(1-theDataSet.modelprofiles.length));
             ddeltax = Math.max((dterm1+dterm2)/dterm3,(dterm1-dterm2)/dterm3);
	  }
          else
	  {
             ddeltax = dmainwidth/(2*dunitcount);
	  }

          //need to handle negative sqrt case
          int ncols = Math.max((int) Math.ceil(dmainwidth/(dunitcount*ddeltax)-1),1);
          double drecwidth = ddeltax*(dunitcount-1);
	  double drecheight = ddeltax*(dunitcount-1);

          canvas = getCanvas();
          canvas.getLayer().removeAllChildren();
 
          String szordertext;
          PText sortText;

          String szpretext;
	  if (theDataSet.bkmeans)
	  {
             szpretext = "Clusters ";
	  }
          else if (theDataSet.tga.bcluster)
	  {
	     szpretext = "Clusters and then profiles "; 
	  }
          else
	  {
             szpretext = "Profiles ";
	  }

          if (theDataSet.tga.szSelectedGO == null)
	  { 
              if (theDataSet.tga.szsortcommand.equals("define"))
	      {
	         szordertext = szpretext+
                                "ordered based on the actual size based p-value gene enrichment of "
                                +"genes in the query gene set";
	      }
              else if (theDataSet.tga.szsortcommand.equals("expdefine"))
	      {
	         szordertext = szpretext+
                                "ordered based on the expected size based p-value gene enrichment of "
                                +"genes in the query gene set";
	      }
              else if ((theDataSet.tga.szsortcommand.equals("id"))||
                       (theDataSet.tga.szsortcommand.equals("default"))&&(theDataSet.bkmeans))
	      {
       	         szordertext = szpretext+"ordered based on "+(theDataSet.bkmeans ?"cluster":"profile")+" ID";
	      }
              else if (theDataSet.tga.szsortcommand.equals("num"))
	      {
	         szordertext = szpretext+"ordered based on the number of genes assigned";
	      }
              else if (theDataSet.tga.szsortcommand.equals("sig"))
	      {
	         szordertext = szpretext+"ordered based on the p-value significance of number of "+
                              "genes assigned versus expected";
	      }
              else if (theDataSet.tga.szsortcommand.equals("exp"))
	      {
	         szordertext = szpretext+"ordered based on the expected number of genes based on "+
                                           "a permutation test";
	      }
	      else 
	      {
	         szordertext = "Clusters ordered based on number of genes and "+
                                 "profiles ordered by significance (default)";
	      }
              sortText = new PText(szordertext);
              sortText.translate(BUFFERLEFT,2);
	   }
           else
	   {
              if (theDataSet.tga.szsortcommand.equals("expgo"))
	      {
	         szordertext = szpretext+"ordered based on the expected size based p-value of gene enrichment of ";
	      }
              else
	      {
		 szordertext = szpretext+"ordered based on the actual size based p-value of gene enrichment of ";
	      }
              szordertext = szordertext +  theDataSet.tga.szSelectedGO+
                          " ("+ ((GoAnnotations.Rec)theDataSet.tga.htGO.get(theDataSet.tga.szSelectedGO)).sztermName+") genes";
              sortText = new PText(szordertext); 
              sortText.translate(BUFFERLEFT,2);
	   }

           sortText.setFont(new Font("times",Font.PLAIN,14));
	   canvas.getLayer().addChild(sortText);

	   float frecwidth = (float) 114.0; 
	   int OFFSET = 100;
	   int EXTRASPACE = 5;

           PNode allButton = PPath.createRectangle((float) 0.0,(float) 0.0,(float) frecwidth,(float) 18.0);
           allButton.translate(2*(SCREENWIDTH/7+EXTRASPACE)-OFFSET,SCREENHEIGHT-65);
           PText theText = new PText("Main Gene Table");
           theText.setFont(new Font("times",Font.PLAIN,12));
           theText.translate(10,2);
           allButton.setPaint(ST.buttonColor);
           allButton.addChild(theText);    

           PNode filteredButton = PPath.createRectangle((float) 0.0,(float) 0.0,(float) frecwidth,(float) 18.0);
           filteredButton.translate(SCREENWIDTH/7+EXTRASPACE-OFFSET,SCREENHEIGHT-65);
           PText thefilteredText = new PText("Filtered Gene List");
           thefilteredText.setFont(new Font("times",Font.PLAIN,12));
           thefilteredText.translate(8,2);
           filteredButton.setPaint(ST.buttonColor);
           filteredButton.addChild(thefilteredText);         
	   canvas.getLayer().addChild(filteredButton);     

           PNode geneplotButton = PPath.createRectangle((float) 0.0,(float) 0.0,(float) frecwidth,(float) 18.0);
           geneplotButton.translate(3*(SCREENWIDTH/7+EXTRASPACE)-OFFSET,SCREENHEIGHT-65);
           PText thegeneText = new PText("Interface Options...");
           thegeneText.setFont(new Font("times",Font.PLAIN,12));
           thegeneText.translate(6,2);
           geneplotButton.setPaint(ST.buttonColor);
           geneplotButton.addChild(thegeneText); 
	   canvas.getLayer().addChild(geneplotButton); 

           PNode sortButton = PPath.createRectangle((float) 0.0,(float) 0.0,(float) frecwidth,(float) 18.0);
           sortButton.translate(4*(SCREENWIDTH/7+EXTRASPACE)-OFFSET,SCREENHEIGHT-65);
           PText thesortText;
	   if (theDataSet.bkmeans)
	   {
              thesortText = new PText("Order Clusters By...");
              thesortText.translate(1,2);
	   }
	   else
	   {
              thesortText = new PText("Order Profiles By...");
	      thesortText.translate(3,2);
	   }

           thesortText.setFont(new Font("times",Font.PLAIN,12));

           sortButton.setPaint(ST.buttonColor);
           sortButton.addChild(thesortText);   
	   canvas.getLayer().addChild(sortButton);
           final MAINGUI2 thisFrame = this;

	   if (!theDataSet.bkmeans)
	   {
              PNode sortClusterButton = PPath.createRectangle((float) 0.0,(float) 0.0,(float) frecwidth,(float) 18.0);
              sortClusterButton.translate(5*(SCREENWIDTH/7+EXTRASPACE)-OFFSET,SCREENHEIGHT-65);
              PText thesortClusterText = new PText("Order Clusters By...");
              thesortClusterText.setFont(new Font("times",Font.PLAIN,12));
              thesortClusterText.translate(2,2);
              sortClusterButton.setPaint(ST.buttonColor);
              sortClusterButton.addChild(thesortClusterText);   
	      canvas.getLayer().addChild(sortClusterButton);

    	      sortClusterButton.addInputEventListener(new PBasicInputEventHandler() 
              {
	         public void mousePressed(PInputEvent event) 
                 {
                    javax.swing.SwingUtilities.invokeLater(new Runnable() 
                    {
                       public void run() 
                       {
		          if (clustersortframe == null)
		          {
                             clustersortframe = new JFrame("Order Clusters and then Profiles by:");
		             clustersortframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                             clustersortframe.setLocation(400,300);
                             theDataSet.tga.bcluster = true;
                             SortTable newContentPane = new SortTable(clustersortframe,thisFrame,
                                                                  theDataSet.tga,theDataSet);
     
                             newContentPane.setOpaque(true); //content panes must be opaque
                             clustersortframe.setContentPane(newContentPane);

                             //Display the window.
                             clustersortframe.pack();
			  }
		          else
		          {
		             clustersortframe.setExtendedState(Frame.NORMAL);
		          }
                          clustersortframe.setVisible(true);
                          //repaint();//drawmain();
		       }
		     });
	          }
	       });   
	   }

           PNode compareButton = PPath.createRectangle((float) 0.0,(float) 0.0,(float) frecwidth,(float) 18.0);
           PImage helpButton = new PImage(Util.getImageURL("Help24.gif"));
	   PImage saveButton = new PImage(Util.getImageURL("Save24.gif"));

	   if (theDataSet.bkmeans)
	   {
	       compareButton.translate(5*(SCREENWIDTH/7+EXTRASPACE)-OFFSET,SCREENHEIGHT-65);
	       saveButton.translate(5*(SCREENWIDTH/7+EXTRASPACE)+20,SCREENHEIGHT-68);
	       helpButton.translate(5*(SCREENWIDTH/7+EXTRASPACE)+50,SCREENHEIGHT-68);
	   }
	   else
	   {
	       compareButton.translate(6*(SCREENWIDTH/7+EXTRASPACE)-OFFSET,SCREENHEIGHT-65);
	       saveButton.translate(6*(SCREENWIDTH/7+EXTRASPACE)+20,SCREENHEIGHT-68);
	       helpButton.translate(6*(SCREENWIDTH/7+EXTRASPACE)+50,SCREENHEIGHT-68);
	   }
	   canvas.getLayer().addChild(helpButton);
	   canvas.getLayer().addChild(saveButton);


           PText compareText = new PText("Compare...");
           compareText.setFont(new Font("times",Font.PLAIN,12));
           compareText.translate(23,2);
           compareButton.setPaint(ST.buttonColor);
           compareButton.addChild(compareText);   
	   canvas.getLayer().addChild(compareButton);

	   final MAINGUI2 thefmaingui2 = this;
	   saveButton.addInputEventListener(new PBasicInputEventHandler()
	   {
	       public void mousePressed(PInputEvent event)
	       {
                  try
                  {
		      int nreturnVal = Util.theChooser.showSaveDialog(thefmaingui2);
                     if (nreturnVal == JFileChooser.APPROVE_OPTION) 
                     {
                        File f = Util.theChooser.getSelectedFile();
			String szext = "";
			String szname = f.getName();
			int nindex = szname.lastIndexOf('.');
		        if (nindex > 0 && nindex < szname.length() - 1) 
                        {
			   szext = szname.substring(nindex + 1).toLowerCase(Locale.ENGLISH);
		        }

			if (szext.equalsIgnoreCase("svg")) 
			{
			   // Get a DOMImplementation.
			   DOMImplementation domImpl = GenericDOMImplementation
									.getDOMImplementation();
			   // Create an instance of org.w3c.dom.Document.
			   String svgNS = "http://www.w3.org/2000/svg";
			   Document document = domImpl.createDocument(svgNS,"svg", null);
			   // Create an instance of the SVG Generator.
			   SVGGraphics2D svgGenerator = new SVGGraphics2D(document);

			   svgGenerator.setColor(Color.white);

			   canvas.paintComponent(svgGenerator);
							// Ask the test to render into the SVG Graphics2D
							// implementation.
							// TestSVGGen test = new TestSVGGen();
							// test.paint(svgGenerator);
							// Finally, stream out SVG to the standard output
							// using
							// UTF-8 encoding.
			    boolean useCSS = true; // we want to use CSS style
													// attributes
			    Writer out = new OutputStreamWriter(new FileOutputStream(f), "UTF-8");
			    svgGenerator.stream(out, useCSS);
			    

			}
			else
			{
                           PrintWriter pw = new PrintWriter(new FileOutputStream(f));       
                           printDefaults(pw);		              
                           pw.close();
			}
		     }
	          }           
                  catch (IOException ex) //FileNotFoundException ex)
                  {
		      //final FileNotFoundException fex = ex;
                     final IOException fex = ex;
                     javax.swing.SwingUtilities.invokeLater(new Runnable() 
                     {
                        public void run() 
                        {
                           JOptionPane.showMessageDialog(null, fex.getMessage(), 
                                "Exception thrown", JOptionPane.ERROR_MESSAGE);
	                }
	             });
                     ex.printStackTrace(System.out);
		  } 
	       }
	    });
			       
	    helpButton.addInputEventListener(new PBasicInputEventHandler() 
            {
	       public void mousePressed(PInputEvent event) 
               {
                  javax.swing.SwingUtilities.invokeLater(new Runnable() 
                  {
                     public void run() 
                     {
                        JDialog helpDialog = new JDialog(thisFrame, "Help", false);
                        Container theHelpDialogPane = helpDialog.getContentPane();
       
                        helpDialog.setBackground(Color.white);
                        theHelpDialogPane.setBackground(Color.white);
		        String szMessage;
		        if (!theDataSet.bkmeans)
		        {
		           szMessage = "The above image diagrams a model temporal expression profile.  "+
                            "The main screen contains all the model expression profiles.  "+
                            "Genes are assigned to the model profile that they most closely match.  "+
                            "The number of genes expected to be assigned to each profile is also computed "+
                            "based on a permuation test.  Model profiles "+
                            "which have a statistically significant number of genes assigned to it as "+
                            "compared to what was expected "+
                            "are colored non-white.  "+
                            "Significant model profiles which are similar to each other are grouped together as a "+
                            "cluster of profiles, and are given the same color.\n\n"+
                            "Clicking on a profile box displays more information about the profile including a graph of "+
                            "the time series of the genes assigned to the profile.  "+
                            "One can also see "+
                            "a list of genes assigned to the profile, or perform a GO gene enrichment analysis on "+
                            "the set of genes assigned.\n"+
                            "In addition to clicking on any profile, "+
                            "there are seven additional buttons on the main window: \n"+
                            "*'Main Gene Table' - displays a table containing the names of all genes that passed filter,"+
                            " their expression values, and model profile assignment.\n"+
                            "*'Filtered Gene List' - contains a list of all the genes that were filtered.\n"+
                            "*'Interface Options...' - Display a window to adjust various interface options.\n"+
                            "*'Order Profiles By...' -  Reorders profiles by enrichment of genes from a selected "+
                            "GO Category or user defined query set of "+
                            "genes, number of genes assigned or expected, significance of the profile, or ID number.  "+
                            "Displays additional information in the profile box based on the reordering chosen.\n"+
		            "*'Order Clusters By...' - Reorders the cluster of profiles based on enrichment of genes "+
                            "from a selected GO Category or a query set.  The p-value the enrichment for "+
		            "the cluster appears in the upper-right hand corner of any profile of the cluster.\n"+
		            "*'Compare...' - Allows one to specify a data set for a time series "+
		            "of the same length to compare against the current data set.  Profile pairs, one from this "+
		            "experiment and one from the comparison experiment, which have a significant overlap in genes "+
                            "assigned to them will be identified in a new window.  A display of all profiles in the "+
                            "comparison experiment will also appear in a new window. \n"+
                            "*Disk icon - This opens a window to specify a file to save the settings used to produce "+
                            "the current analysis to a specified file. "+
                            " The settings can be opened using the 'Load Saved Settings' on the main interface or "+
                            "from the command line as a defaults file.\n\n"+ 
                            "Note also that the main interface is zoomable and pannable. "+
                            "Hold the right button down to zoom or the left to pan while moving the mouse.";
			}
		        else
		        {
		           szMessage = "The above image contains a legend for a K-means cluster box.  "+
                            "The main screen contains boxes for all K-means cluster.  "+
                            "Clicking on a K-means cluster box displays more information about the cluster "+
                            "including a graph of "+
                            "the time series of the genes belonging to the cluster.  "+
                            "One can also see "+
                            "a list of genes belonging to the cluster, or perform a GO gene enrichment analysis on "+
                            "the set of genes.\n"+
                            "In addition to clicking on any cluster box, "+
                            "there are five additional buttons on the main window: \n"+
                            "*'Main Gene Table' - displays a table containing the names of all genes that passed filter,"+
                            " their expression values, and cluster assignment.\n"+
                            "*'Filtered Gene List' - contains a list of all the genes that were filtered.\n"+
                            "*'Interface Options...' - Display a window to adjust various interface options.\n"+
                            "*'Order Clusters By...' -  Reorders clusters by enrichment of genes from a selected "+
                            "GO Category or user defined query set of "+
                            "genes, number of genes assigned  or ID number.  "+
                            "Displays additional information in the cluster box based on the reordering chosen.\n"+
		            "*'Compare...' - Allows one to specify a data set for a time series "+
		            "of the same length to compare against the current data set. Cluster pairs, one from this "+
		            "experiment and one from the comparison experiment, which have a significant overlap in genes "+
                            "assigned to them will be identified in a new window.  A display of all cluster in the "+
                            "comparison experiment will also appear in a new window. \n\n "+
                            "Note also that the main interface is zoomable and pannable. "+
                            "Hold the right button down to zoom or the left to pan while moving the mouse.";
			}

                        JTextArea textArea = new JTextArea(szMessage,10,70);
                        textArea.setLineWrap(true);
                        textArea.setWrapStyleWord(true);
                        textArea.setBackground(Color.white);
                        textArea.setEditable(false);

         	        ImageIcon ii;
	 	        if (theDataSet.bkmeans)
		        {
                           ii = Util.createImageIcon("p4.png");
	   	        }
		        else
		        {
                           ii = Util.createImageIcon("p42.png");
	  	        }
                        JLabel jl = new JLabel(ii);
         
		        JPanel psl = new JPanel();
		        psl.setLayout(new SpringLayout());
	 	        psl.setBackground(Color.white);
		        psl.add(jl);
		        JScrollPane jsp2 = new JScrollPane(textArea);
		        psl.add(jsp2);
                        SpringUtilities.makeCompactGrid(psl,2,1,0,2,0,0);
	          
			JScrollPane jsp =new JScrollPane(psl);

			theHelpDialogPane.add(jsp);
			theHelpDialogPane.setSize(820,600);
			theHelpDialogPane.validate();

			helpDialog.setLocation(thisFrame.getX()+50,thisFrame.getY()+25);

			helpDialog.setSize(820,600);
			helpDialog.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
			helpDialog.setVisible(true);
		     }
		  });
	       }
	   });   

           ///////////////////////////////////////////////////////////////
	   
	   compareButton.addInputEventListener(new PBasicInputEventHandler() 
           {
	      public void mousePressed(PInputEvent event) 
              {
                 theOptions.setVisible(true);   
	      }
	   });   

	   ///////////////////////////////////////////////////////////////
	     
	   geneplotButton.addInputEventListener(new PBasicInputEventHandler() 
           {
	      public void mousePressed(PInputEvent event) 
              {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                    public void run() 
                    {
		       if (thegeneplotframe != null)
		       {
                          thegeneplotframe.setExtendedState(Frame.NORMAL);
                          thegeneplotframe.setVisible(true);
		       }
		    }
		  });
	      }
	   }); 


	   //////////////////////////////////////////////////////////////

	   sortButton.addInputEventListener(new PBasicInputEventHandler() 
           {
	      public void mousePressed(PInputEvent event) 
              {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                    public void run() 
                    {
		       if (profilesortframe == null)
		       {
		          if (theDataSet.bkmeans)
		          {
                             profilesortframe = new JFrame("Order Clusters by:");
		          }
		          else
			  {
                             profilesortframe = new JFrame("Order Profiles by:");
		          }
		          profilesortframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                          profilesortframe.setLocation(400,300);
                          theDataSet.tga.bcluster = false;
                          SortTable newContentPane = new SortTable(profilesortframe,thisFrame,
                                                        theDataSet.tga,theDataSet);
     
                          newContentPane.setOpaque(true); //content panes must be opaque
                          profilesortframe.setContentPane(newContentPane);

                          //Display the window.
                          profilesortframe.pack();
		       }
		       else
		       {
		          profilesortframe.setExtendedState(Frame.NORMAL);
		       }
                       profilesortframe.setVisible(true); 
		    }
		 });
	      }
	   });   
	   

	   filteredButton.addInputEventListener(new PBasicInputEventHandler() 
           {
	      public void mousePressed(PInputEvent event) 
              {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                    public void run() 
                    {
	
                       JFrame frame = new JFrame("Table of Genes Filtered");
	               frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                       frame.setLocation(20,50);
                       FilteredTable newContentPane = new FilteredTable(frame,theDataSet);
                       newContentPane.setOpaque(true); //content panes must be opaque
                       frame.setContentPane(newContentPane);

                       //Display the window.
                       frame.pack();
                       frame.setVisible(true);
		    }
		  });
	       }
	   });


	   allButton.addInputEventListener(new PBasicInputEventHandler() 
           {
	      public void mousePressed(PInputEvent event) 
              {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                   public void run() 
                   {
                      JFrame frame = new JFrame("Table of Genes Passing Filter");
	              frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                      frame.setLocation(20,50);

                      MainTable newContentPane = new MainTable(frame,theDataSet,thegeneplotpanel,cf);
                      newContentPane.setOpaque(true); //content panes must be opaque
                      frame.setContentPane(newContentPane);

                     //Display the window.
                     frame.pack();
                     frame.setVisible(true);
		   }
		 });
	       }
	   });

	   canvas.getLayer().addChild(allButton);
     
           if (bsaveprofile)
	   {
              profilenodes = new PNode[profilelookup.length];
	   }

           int NOLDUSED= NUSED;
           boolean bmakecolors;
           if (savedcolors == null)
	   {
              savedcolors = new Color[profilelookup.length];
              bmakecolors = true;
	   }
	   else
	   {
	       bmakecolors = false;
	   }

	   double profilek[] = new double[theDataSet.modelprofiles.length];
	   double dk = 0;
	   for (int nprofile = 0; nprofile < theDataSet.modelprofiles.length; nprofile++)
	   {
	       profilek[nprofile] = 0;
	       for (int ncurrindex = 0; ncurrindex < theDataSet.modelprofiles[nprofile].length; ncurrindex++)
	       {
                  double dtempk =Math.abs(theDataSet.modelprofiles[nprofile][ncurrindex]); 
	   	  if (dtempk> dk)
		  {
		     dk = dtempk;
		  }
                  if (dtempk > profilek[nprofile])
	          {
		     profilek[nprofile] = dtempk;
		  }
	       }

               profilek[nprofile] = (theDataSet.modelprofiles[0].length-1)/profilek[nprofile];
	   }
	   dk = (theDataSet.modelprofiles[0].length-1)/dk;
	   
           //computes global genek
	   double dgenek = 0;
	   if ((ngenescale == 2)&&(ngenedisplay >= 1))
	   {
	      //genes should currently be displayed
	      //on global gene scale
              for (int nprofile = 0; nprofile < profilelookup.length; nprofile++)
              {
	         //for global genek
                 int nactualprofile = profilelookup[nprofile]; 
                 ArrayList currProfileAssignments = theDataSet.profilesAssigned[nactualprofile];
                 int numuniquegenes = currProfileAssignments.size(); 

	         for (int nindex =0; nindex < numuniquegenes; nindex++)
	         {
                    int nrow =  ((Integer) currProfileAssignments.get(nindex)).intValue();
                    if ((!bscalevisible)||(ngenedisplay==2)||
                        (theDataSet.tga.isOrder(theDataSet.genenames[nrow])))
		    {
                        //not scaling visible
                        //or displaying all
			//or ordering by it and thus selected
                       for (int ncol = 0; ncol < theDataSet.data[nrow].length; ncol++)
	               {
		          if ((theDataSet.pmavalues[nrow][ncol]>0)&&
                              (Math.abs(theDataSet.data[nrow][ncol]) > dgenek))
		          {
		             dgenek = Math.abs(theDataSet.data[nrow][ncol]);
			  }
		       }
	            }
	         }

		 if (theDataSet.bkmeans)
		 {
		    for (int ncol = 0; ncol < theDataSet.modelprofiles[nprofile].length; ncol++)
		    {
	               dgenek = Math.max(dgenek,theDataSet.modelprofiles[nprofile][ncol]);
                    }     
		 }
	      }


	      if (dgenek > 0)
	      {
	         dgenek = (theDataSet.data[0].length-1)/dgenek;
              }
	   }

	   //figure out X-axis
           theDataSet.dwidthunitsCum = new double[theDataSet.dsamplemins.length];
           theDataSet.dwidthunitsCum[0] = 0;
      
	   boolean bincreasing = true;
     
           try
           {
              double dprev = Util.removeUnits(theDataSet.dsamplemins[0]);
		   
              for (int ni = 0; (ni < theDataSet.dwidthunitsCum.length-1)&&(bincreasing); ni++)
              {
	         double dnext = Util.removeUnits(theDataSet.dsamplemins[ni+1]);
	         theDataSet.dwidthunitsCum[ni+1] =
                                theDataSet.dwidthunitsCum[ni] + (dnext-dprev);
		 if (dnext < dprev)
	      	 {
		    bincreasing = false;
	            buniformXaxis = true;
	       	    binvalidreal = true;
	         }
	         dprev = dnext;
	      }
	           
              double dtotal = theDataSet.dwidthunitsCum[theDataSet.dwidthunitsCum.length-1]; 
	      for (int ni = 1; ni < theDataSet.dwidthunitsCum.length; ni++)
	      {
	         theDataSet.dwidthunitsCum[ni]= theDataSet.dwidthunitsCum[ni]/dtotal;
	      }
	   }
	   catch (IllegalArgumentException iae)
	   {
              for (int ni = 1; ni < theDataSet.dwidthunitsCum.length; ni++)
	      {
	         theDataSet.dwidthunitsCum[ni]= ni/(double)(theDataSet.dsamplemins.length-1);
	      }
	      buniformXaxis = true;
	      binvalidreal = true;
	   }
	    
	   if ((buniformXaxis)||(!bincreasing))
           { 
	      for (int ni = 1; ni < theDataSet.dwidthunitsCum.length; ni++)
	      {
	          theDataSet.dwidthunitsCum[ni]= ni/(double) (theDataSet.dsamplemins.length-1);
	      }
	   }

           for (int nprofile = 0; nprofile < profilelookup.length; nprofile++)
           {
              node = PPath.createRectangle(0,0,(float)drecwidth,(float)drecheight);
	       
              double dstartx = BUFFERLEFT + dunitcount *ddeltax*(nprofile%ncols);
              double dstarty = BUFFERTOP + dunitcount *ddeltax*(nprofile/ncols);
            
              int nactualprofile = profilelookup[nprofile];  
 
              if  (bmakecolors)
	      {
	         Color currColor;
                 if (cluster[nactualprofile] == numclusters)
                 {
	            currColor = new Color((float) 1, (float) 1, (float) 1, (float) .4);
                    node.setPaint(currColor);
                 }
                 else if (cluster[nactualprofile]+NOLDUSED >= clustercolors.length)
	         {
	            while (cluster[nactualprofile]-clustercolors.length+NOLDUSED>= coloroverflow.size())
		    {
	               NUSED++;
	               coloroverflow.add(new Color((float) (Math.random()*.8+.05), (float) (Math.random()*.8+.05), 
						    (float) (Math.random()*.8+.05),(float)0.4));
	            }
	            currColor =(Color) coloroverflow.get(cluster[nactualprofile]+NOLDUSED -clustercolors.length);
                    node.setPaint(currColor);
	         }
	         else
                 {
	            if (cluster[nactualprofile]+NOLDUSED >= NUSED)
		    {
                       NUSED = cluster[nactualprofile]+NOLDUSED+1;
	            }
                    currColor = clustercolors[cluster[nactualprofile]+NOLDUSED];
                    node.setPaint(currColor);
	          }
                  savedcolors[nactualprofile] = currColor;
	       }
               else
	       {
                  node.setPaint(savedcolors[nactualprofile]);
	       }

               float[] xp = {0,0};
               float[] yp = {0,0};

	       //////////////////////////////////
	       //this plots individual genes on main window
	       ////////////////////////////////
               ArrayList currProfileAssignments = theDataSet.profilesAssigned[nactualprofile];
               int numuniquegenes = currProfileAssignments.size();
	       if ((ngenescale == 1)&&(ngenedisplay >= 1))
	       {
		   //profile or cluster specific and display geens
	          dgenek = 0;
	          for (int nindex =0; nindex < numuniquegenes; nindex++)
	          {
                     int nrow =  ((Integer) currProfileAssignments.get(nindex)).intValue();
		     if ((ngenedisplay==2)||(!bscalevisible) ||
			    (theDataSet.tga.isOrder(theDataSet.genenames[nrow])))
		     {
			 //always display
			 //don't care if visible
			 //or is visible
                        for (int ncol = 0; ncol < theDataSet.data[nrow].length; ncol++)
	                {
		           if ((theDataSet.pmavalues[nrow][ncol]>0)&&
                               (Math.abs(theDataSet.data[nrow][ncol]) > dgenek))
		           {
		              dgenek = Math.abs(theDataSet.data[nrow][ncol]);
		           }
			}
	             }
	          }
		  if (dgenek > 0)
		  {
	             dgenek = (theDataSet.data[0].length-1)/dgenek;
		  }
	       }

	       double dfactor;
	       if (ngenescale >=1)
	       {
		   //want to use maximize profile box space if not doing gene specific scaling
	       	 dfactor = 1;
	       }
	       else
	       {
	       	 dfactor = .75;
	       }

		  
	       boolean bneedgenetick = false;    
	       if ((ngenedisplay >= 1)&&((!theDataSet.bkmeans)||(ngenescale>=1)))
	       {
		   //potentially display some genes
	          for (int nindex =0; nindex < numuniquegenes; nindex++)
	          {
                     int nrow =  ((Integer) currProfileAssignments.get(nindex)).intValue();
	             if ((ngenedisplay==2)||(theDataSet.tga.isOrder(theDataSet.genenames[nrow])))
	             {
			 //should show gene
		        if (ngenescale == 0)
		        {
			    //individual scale
		           dgenek = 0;
                           for (int ncol = 0; ncol < theDataSet.data[nrow].length; ncol++)
	                   {
		              if ((theDataSet.pmavalues[nrow][ncol]>0)&&
                                  (Math.abs(theDataSet.data[nrow][ncol]) > dgenek))
		              {
		                 dgenek = Math.abs(theDataSet.data[nrow][ncol]);
			      }
		           }
			   if (dgenek > 0)
			   {
 	                      dgenek = (theDataSet.data[0].length-1)/dgenek;
			   }
		        }
			else
			{
			    bneedgenetick = true;
			}

	                for (int ncol = 0; ncol < theDataSet.data[0].length-1; ncol++)
	                {
		           if ((theDataSet.pmavalues[nrow][ncol]>0)&&(theDataSet.pmavalues[nrow][ncol+1]>0))
	                   {
			      xp[0] = (float) (ddeltax*theDataSet.dwidthunitsCum[ncol]*(theDataSet.data[0].length-1)); 
                              xp[1] = (float) (ddeltax*theDataSet.dwidthunitsCum[ncol+1]*(theDataSet.data[0].length-1));
                              yp[0] = (float) (ddeltax/2*
		  		   (theDataSet.numcols-1-dfactor*
		  		    dgenek*theDataSet.data[nrow][ncol]));
                              yp[1] = (float) (ddeltax/2*
		  		      (theDataSet.numcols-1-dfactor*
		  		      dgenek*theDataSet.data[nrow][ncol+1]));
                              PPath line = PPath.createPolyline(xp,yp);                
                              line.setStrokePaint(thegenecolor);
                              node.addChild(line);
			   }
	                }		     
	             }
	          }
 	       }


	       if (((bneedgenetick)||(theDataSet.bkmeans))&&(bdisplaymaintick))
	       {
		   double dtickk;
		   if (theDataSet.bkmeans)
		   {
		      //this one for k-means euclidean
		      if ((ngenescale==0)||(dgenek==0))
		      {
			  dtickk = dk;
		      }
                      else
		      {
			  dtickk = dgenek;
		      }
		   }
		   else
		   {
		       dtickk = profilek[nactualprofile];
		   }

		   if (!bneedgenetick&&theDataSet.bkmeans)
		   {
		       dtickk = dk;
		   }
		   else
		   {
		       dtickk = dgenek;
		   }
		   boolean bcontinue = true;
		   double dunit = 0;
		   while (bcontinue)
		   {
	              yp[0] = (float) (ddeltax/2*
		  	      (theDataSet.numcols-1-dfactor*dtickk*dunit));

		      if (yp[0]<0)
		      {
			  bcontinue = false;
		      }
		      else
		      {
	                 xp[0] = 0;
			 if (dunit == 0)
			 {
	                    xp[1] =(float)(0.2*drecwidth);
			 }
			 else
			 {
	                    xp[1] =(float)(0.1*drecwidth);
			 }

	                 yp[1] = yp[0];
	                 PPath tickmark = PPath.createPolyline(xp,yp);
	                 tickmark.setStrokePaint(Color.black);
	                 node.addChild(tickmark);
	                 tickmark.setVisible(true);

			 yp[0] = (float) (ddeltax/2*
		  	      (theDataSet.numcols-1+dfactor*dtickk*dunit));
			 yp[1] = yp[0];
	                 PPath tickmarkneg = PPath.createPolyline(xp,yp);
	                 tickmarkneg.setStrokePaint(Color.black);
	                 node.addChild(tickmarkneg);
	                 tickmarkneg.setVisible(true);
		      }
		      dunit += dtickintervalmain;
		   }
	       }

	       if (bdisplayprofileline)
	       {
                  double[] currprofile = theDataSet.modelprofiles[nactualprofile];

                  for (int npoint = 0; npoint < theDataSet.numcols; npoint++)
	          {
                     xp[0] = xp[1]; 
                     yp[0] = yp[1];
                     xp[1] = (float) (ddeltax*theDataSet.dwidthunitsCum[npoint]*(theDataSet.data[0].length-1));

		     if (theDataSet.bkmeans)
		     {
		        //this one for k-means euclidean
		        if ((ngenescale==0)||(dgenek==0))
		        {
                           yp[1] = (float) (ddeltax/2*
				      (theDataSet.numcols-1-dk*currprofile[npoint]));
		        }
                        else
		        {
                           yp[1] = (float) (ddeltax/2*
				      (theDataSet.numcols-1-dgenek*currprofile[npoint]));		  
		        }
		     }
		     else
		     {
                        yp[1] = (float) (ddeltax/2*
		  		   (theDataSet.numcols-1-.75*
		  		    profilek[nactualprofile]*currprofile[npoint]));
		     }

                     if (npoint >=1)
                     {
                        PPath line = PPath.createPolyline(xp,yp);                
                        line.setStrokePaint(Color.black);
		        //option here to hide model profile
		     
		        line.setVisible(true);
                        node.addChild(line);
		     }
                  }
	       }


	       if (thegeneplotframe == null)
	       {
                  thegeneplotframe = new JFrame("Interface Options");
                  thegeneplotframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                  thegeneplotframe.setLocation(400,150);
                  thegeneplotpanel = new GenePlotPanel(thegeneplotframe,thisFrame,theDataSet.bkmeans);
                  thegeneplotpanel.setOpaque(true); //content panes must be opaque
                  thegeneplotframe.setContentPane(thegeneplotpanel);
                  //Display the window.
                  thegeneplotframe.pack();
	          thegeneplotframe.setVisible(false);
	       } 

	       String szLabel = ""+nactualprofile;

	       PText text;
	        if (bdisplayID)
	        {
                   text = new PText(szLabel);
                   text.setFont(new Font("times",Font.PLAIN,(int) (.85*(theDataSet.numcols/5.0)
                                                                         *Math.ceil(ddeltax))));
		  //option here to hide profile ID
		  text.setVisible(true);

                  text.translate(ddeltax/8,-1);
                  node.addChild(text);
		}

		if (bdisplaydetail)
		{
		   profilenodes[profilelookup[nprofile]] =(PNode) node.clone();   

		   if (((theDataSet.tga.szsortcommand.equals("go"))||
                       (theDataSet.tga.szsortcommand.equals("define")))&&
                       (theDataSet.tga.bcluster)&& (theDataSet.tga.tpgr[nprofile].dclusterpval<1.5))
	           {
		      szLabel = ""+doubleToSz(theDataSet.tga.tpgr[nprofile].dclusterpval);  
                      text = new PText(szLabel);
                      text.setFont(new Font("times",Font.PLAIN,(int) (0.8*(theDataSet.numcols/5.0)
                                                                  *Math.ceil(ddeltax))));

                      double dleft = drecwidth- szLabel.length()/2.0*
		                     ((int) (0.8*(theDataSet.numcols/5.0) *Math.ceil(ddeltax)));

                      text.translate(dleft,0);
                      node.addChild(text); 
		      text.setVisible(true);
		   }


                   if ((theDataSet.tga.szsortcommand.equals("expgo"))||
                     (theDataSet.tga.szsortcommand.equals("go"))||
                     (theDataSet.tga.szsortcommand.equals("expdefine"))||
                     (theDataSet.tga.szsortcommand.equals("define")))

                   {
                      szLabel = ""+(int)Math.round(theDataSet.tga.tpgr[nprofile].dgenes)+";"
                                             +doubleToSz(theDataSet.tga.tpgr[nprofile].dpval);  
                      text = new PText(szLabel);
                      text.setFont(new Font("times",Font.PLAIN,(int) (0.8*(theDataSet.numcols/5.0)*Math.ceil(ddeltax))));
                      text.translate(ddeltax/9,drecheight-ddeltax*(theDataSet.numcols/5.0));
                      node.addChild(text);
		      text.setVisible(true);
		   }


                   if (theDataSet.tga.szsortcommand.equals("sig"))
		   {
		      szLabel = ""+doubleToSz(theDataSet.pvaluesassignments[nactualprofile]);             
                      text = new PText(szLabel);
                      text.setFont(new Font("times",Font.PLAIN,(int) (0.8*(theDataSet.numcols/5.0)
                                                                     *Math.ceil(ddeltax))));
                      text.translate(ddeltax/9,drecheight-ddeltax*(theDataSet.numcols/5.0));
                      node.addChild(text);     
		      text.setVisible(true);
		   }

                  if (theDataSet.tga.szsortcommand.equals("exp"))
		  {
		     szLabel = ""+(int)Math.round(theDataSet.expectedassignments[nactualprofile]);             
                     text = new PText(szLabel);
                     text.setFont(new Font("times",Font.PLAIN,(int) (0.8*(theDataSet.numcols/5.0)
                                                                       *Math.ceil(ddeltax))));
                     text.translate(ddeltax/9,drecheight-ddeltax*(theDataSet.numcols/5.0));
                     node.addChild(text);     
		     text.setVisible(true);
		  }
		
                  if (theDataSet.tga.szsortcommand.equals("num"))
		  {
		     szLabel = ""+(int) Math.round(theDataSet.countassignments[nactualprofile]);             
                     text = new PText(szLabel);
                     text.setFont(new Font("times",Font.PLAIN,(int) (0.8*(theDataSet.numcols/5.0)
                                                                   *Math.ceil(ddeltax))));
                     text.translate(ddeltax/9,drecheight-ddeltax*(theDataSet.numcols/5.0));
                     node.addChild(text);     
		     text.setVisible(true);
		  }
		}
		
                final int nprofilef = nprofile;
		node.addInputEventListener(new PBasicInputEventHandler() 
                {
		   public void mousePressed(PInputEvent event) 
                   {
                      if (event.getButton() == MouseEvent.BUTTON1)
                      {                             
                         javax.swing.SwingUtilities.invokeLater(new Runnable() 
                         {
                            public void run() 
                            {
                               ProfileGui pg;

			       pg = new ProfileGui(theDataSet,profilelookup[nprofilef],ranklookup,
							 null,-1,null,null,null,thegeneplotpanel,cf);

			       openProfiles.add(pg);
                               pg.addWindowListener(new WindowAdapter()
                               {
                                  public void windowClosing(WindowEvent we) 
                                  {
				     int nindexremove = openProfiles.indexOf(we.getSource());
				     if (nindexremove >= 0)
				     {
                                        openProfiles.remove(nindexremove);
				     }
				  }
			       });

                               pg.setLocation(20,50);        
                               pg.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                               pg.pack();
                               pg.setSize(new Dimension(SCREENWIDTH,SCREENHEIGHT));
			       pg.setVisible(true);
			    }
	                 });
		      }
		   }
	       });
               node.translate(dstartx,dstarty);
               canvas.getLayer().addChild(node);
           }
	      
	   bsaveprofile = false;
           for (int nindex = 0; nindex < openProfiles.size(); nindex++)
	   {
	       ((ProfileGui) openProfiles.get(nindex)).repaint();
	   }
	       
           if (thecomparegui != null)
	   {
	      thecomparegui.drawcomparemain();
	   }
       }
    }

    /**
     * Renders the dialog to specify a comparison data set
     */
    public void makeCompareDialog()
    { 
       theOptions = new JDialog(this, "Compare ", false);
       
       Container theDialogContainer = theOptions.getContentPane();
       theDialogContainer.setBackground(Color.white);
       makeComparePanel(theOptions.getContentPane());
       theOptions.pack();
       theOptions.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
       theOptions.setLocation(this.getX(),this.getY()+200); 
  
       theOptions.addWindowListener(new WindowAdapter() 
       {
          public void windowClosing(WindowEvent we) 
          {  
             boolean bok = true;
             try
	     {
	        thespinnersigLevel.commitEdit();
                thespinnernumgenes.commitEdit();
             }
             catch (ParseException ex)
	     {
		 Toolkit.getDefaultToolkit().beep();
		 bok = false;
	     }
                    
             if (bok)
	     {
                theOptions.setVisible(false);
	     }
	  }
       }); 
    }

    /**
     * Renders the window to provide help information
     */
    public  void makeHelpDialog(Component esource)
    {
       JDialog helpDialog = new JDialog(this, "Help", false);
       Container theHelpDialogPane = helpDialog.getContentPane();
       
       helpDialog.setBackground(Color.white);
       theHelpDialogPane.setBackground(Color.white);
       String szMessage = "";
       
       JTextArea textArea = null;

       if (esource == executecompareHButton)
       {
          if (theDataSet.bkmeans)
	  {
             szMessage = "Pressing this button will open two new windows.  One window contains the main "+
                         "K-means clusters interface "+
                         "for the comparison data, while the other window is the main comparison window "+
                         "which shows pairs of clusters one from each experiment with a significant number of genes in their "+
                         "intersection.";
	    }
	    else
	    {
               szMessage = "Pressing this button will open two new windows.  One window contains the model profile "+
                           "overview for the comparison data, while the other window is the main comparison window "+
                           "which shows pairs of profiles one from each experiment with a significant number of genes in their "+
                           "intersection.";
	    }
	}
	if (esource == compare1HButton)
	{
            szMessage = "This field specifies a data file for the comparison experiment.  The data file needs to be"+
                        " in the same format as an original data set file and have the same number of time points.";
	}
        else if (esource == compare2HButton)
	{
            szMessage = "Pressing this button opens a dialog window through which to specify data "+
                        "for the repeat comparison experiments.  The data needs to be"+
                        " in the same format as an original data set file and have the same number of time points.  "+
                  "If the button is yellow then there is currently repeat data loaded, otherwise the button is gray.";
	}
        else if (esource == sigLevelHButton)
	{
	    if (theDataSet.bkmeans)
	    {
               szMessage = "The maximum uncorrected p-value (based on the hypergeometric distribution) "+
                        "of the intersection of "+
                        "genes assigned to two K-means clusters for the intersection to be considered significant.  "+
                        "One of the clusters is from "+
                        "the original data set (the currently displayed data set) and the other "+
                        "cluster comes from the comparison data set being specified.";
	    }
	    else
	    {
               szMessage = "The maximum uncorrected p-value (based on the hypergeometric distribution) "+
                        "of the intersection of "+
                        "genes assigned to two profiles for the intersection to be considered significant.  "+
                        "One of the profiles is from "+
                        "the original data set (the currently displayed data set) and the other "+
                        "profile comes from the comparison data set being specified.";
	    }
	}
        else if (esource == numgenesHButton)
	{
	    if (theDataSet.bkmeans)
	    {
               szMessage = "The minimum number of genes required to be in the intersection of two K-means clusters for "+
                       "the intersection to be considered significant. One of the clusters is from "+
                        "the original data set (the currently displayed data set) and "+
                       "the other clusters comes from the comparison data set being specified.";
	    }
	    else
	    {
               szMessage = "The minimum number of genes required to be in the intersection of two profiles for "+
                       "the intersection to be considered significant. One of the profiles is from "+
                        "the original data set (the currently displayed data set) and "+
                       "the other profile comes from the comparison data set being specified.";
	    }
	}
        else if (esource == viewHButton)
	{
            szMessage = "Pressing the 'View Comparison Data File' displays the contents of the file listed in the "
                       +"above textbox.";
	}
        textArea = new JTextArea(szMessage,7,60);
        textArea.setLineWrap(true);
        textArea.setWrapStyleWord(true);

        textArea.setBackground(Color.white);
        textArea.setEditable(false);
     
        JScrollPane jsp =new JScrollPane(textArea);
        theHelpDialogPane.add(jsp);
        helpDialog.setLocation(theOptions.getX()+50,theOptions.getY()+10);
      
        helpDialog.pack();
        helpDialog.setDefaultCloseOperation(DISPOSE_ON_CLOSE);
        helpDialog.setVisible(true);  
    }

    /**
     * Responds to buttons being pressed
     */
    public void actionPerformed(ActionEvent e) 
    {
        final Object esource = e.getSource();

	if ((esource == compare1HButton)||
            (esource == compare2HButton)||
            (esource == executecompareHButton)||
            (esource == sigLevelHButton)||
	    (esource == viewHButton)||
	    (esource == numgenesHButton))
	{
           javax.swing.SwingUtilities.invokeLater(new Runnable() 
           {
              public void run() 
              {
                 makeHelpDialog((Component) esource);
	      }
           });
	}
        else if (esource == browse1Button) 
        {
           int returnVal = ST.fc.showOpenDialog(this);

           if (returnVal == JFileChooser.APPROVE_OPTION) 
           {
              File file = ST.fc.getSelectedFile();
              compare1Field.setText(file.getAbsolutePath());
           } 
        } 
	else if (esource == viewButton)
	{
	   final String szfile = compare1Field.getText();
	   final JFrame fframe = this;
	   if ((new File(szfile)).exists())
	   {
              javax.swing.SwingUtilities.invokeLater(new Runnable() 
              {
                 public void run() 
                 {
	            DataTable newContentPane = new DataTable(fframe,szfile,false);

                    JFrame dtframe = new JFrame(szfile);
	            dtframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                    dtframe.setLocation(20,50);
                    newContentPane.setOpaque(true); //content panes must be opaque
                    dtframe.setContentPane(newContentPane);
                    //Display the window.
                    dtframe.pack();
                    dtframe.setVisible(true);
		 }
 	      }); 
	   }
	   else
	   {
              javax.swing.SwingUtilities.invokeLater(new Runnable() 
              {
                 public void run() 
                 {
                    JOptionPane.showMessageDialog(fframe, "File '"+szfile+"' was not found.", 
                                                  "Error", JOptionPane.ERROR_MESSAGE);
		 }
	      });
	   }
	}
	else if (esource == repeatButton)
	{
           theCompareRepeatList.setLocation(theOptions.getX()+75,theOptions.getY());
           theCompareRepeatList.setVisible(true);
	}
        else if (esource == compareButtonDialog)
	{
	   String szcompare1val =  compare1Field.getText();          
           String szsigLevel = thespinnersigLevel.getValue().toString();
           String sznumgenes = thespinnernumgenes.getValue().toString();

           try
           {
	       boolean ballselected = theCompareRepeatList.allButton.isSelected();
	       final CompareInfo ci = new CompareInfo(theDataSet, szcompare1val,
			 szsigLevel, sznumgenes, theCompareRepeatList.data, ballselected,cf);

               ci.comparesetfmnel.otherset = theDataSet;
               ci.comparesetfmnel.bothersetorigset = true;
               theDataSet.otherset = ci.comparesetfmnel;
               theDataSet.bothersetorigset = false;

               javax.swing.SwingUtilities.invokeLater(new Runnable() 
               {
                  public void run() 
                  {
                     try
                     {
		        thecomparegui = new CompareGui(ci,profilenodes,thegeneplotpanel,cf);
		        ci.compareframe.thecomparegui = thecomparegui;
		        edu.umd.cs.piccolo.PCanvas.CURRENT_ZCANVAS = null;
	
                        if (profilesortframe != null)
		        {
                           SortTable pt =((SortTable) profilesortframe.getContentPane());
                           if (pt.defineframe != null)
			   {
			      DefineGeneSet dgs = ((DefineGeneSet) pt.defineframe.getContentPane());
                              dgs.selectgenesButton.setEnabled(true); 
			   }
		        }
                        if (clustersortframe != null)
		        {
                           SortTable pt =((SortTable) clustersortframe.getContentPane());
                           if (pt.defineframe != null)
			   {
		 	      DefineGeneSet dgs = ((DefineGeneSet) pt.defineframe.getContentPane());
                              dgs.selectgenesButton.setEnabled(true); 
			   }
		        }

                        thecomparegui.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                        thecomparegui.setLocation(35,80);
		        thecomparegui.setVisible(true);
		     }
                     catch (Exception ex)
                     {
			JOptionPane.showMessageDialog(null, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                        ex.printStackTrace(System.out);
                     } 
		  }
	       });
	       theOptions.setVisible(false);
               theOptions.dispose();
	   }
           catch (Exception ex)
           {
	       JOptionPane.showMessageDialog(null, ex.getMessage(), 
                               "Exception thrown", JOptionPane.ERROR_MESSAGE);
                     ex.printStackTrace(System.out);
           } 	      
	} 
    }

    //////////////////////////////////////////////////////////////////////////////////
    /**
     * Renders the panel within the dialog to compare data sets
     */
    protected void makeComparePanel(Container cp)
    {
       JSpinner.NumberEditor jft = (JSpinner.NumberEditor) thespinnersigLevel.getEditor();                 
       DecimalFormat df = jft.getFormat();
       df.setMaximumFractionDigits(4);
     
       thespinnersigLevel.setPreferredSize(new Dimension(60,24));
       thespinnersigLevel.setMaximumSize(new Dimension(60,24));

       BoxLayout layout = new BoxLayout(cp,BoxLayout.Y_AXIS);
       cp.setLayout(layout);
       cp.setBackground(ST.lightBlue);
       JPanel p = new JPanel(new SpringLayout());

       p.add(compare1Label);
 
       compare1Field.setColumns(ST.NUMCOLS);
       p.add(compare1Field);
       p.add(browse1Button);
       compare1HButton.addActionListener(this);
       p.add(compare1HButton);

       SpringUtilities.makeCompactGrid(p,1,4,6,6,6,6); 
       p.setBackground(ST.lightBlue);
   
       cp.add(p);

       JPanel prepeat = new JPanel(new SpringLayout());
       if (ST.vRepeatCompareDEF.size() >= 1)
       { 
          repeatButton.setBackground(ST.buttonColor);
       }
       prepeat.add(viewButton);
       prepeat.add(viewHButton);
       prepeat.add(repeatButton);
       prepeat.add(compare2HButton);

       repeatButton.addActionListener(this);
       viewButton.addActionListener(this);
       viewHButton.addActionListener(this);
       compare2HButton.addActionListener(this);
       prepeat.setBackground(ST.lightBlue);
       SpringUtilities.makeCompactGrid(prepeat,2,2,6,6,6,6);        
       cp.add(prepeat);
       browse1Button.addActionListener(this);

       JPanel panelParams = new JPanel(new SpringLayout());
       panelParams.setBackground(ST.lightBlue);
       panelParams.add(sigLevelLabel);

       thespinnernumgenes.setPreferredSize(new Dimension(60,24));
       thespinnernumgenes.setMaximumSize(new Dimension(60,24));
       panelParams.add(thespinnersigLevel);
      
       sigLevelHButton.addActionListener(this);
       panelParams.add(sigLevelHButton);
       panelParams.add(numgenesLabel); 

       panelParams.add(thespinnernumgenes);
       numgenesHButton.addActionListener(this);
       panelParams.add(numgenesHButton);
       SpringUtilities.makeCompactGrid(panelParams,2,3,6,6,6,6); 
       cp.add(panelParams);

       JPanel comparepanel = new JPanel();
       comparepanel.setBackground(ST.lightBlue);
       compareButtonDialog.setBackground(ST.buttonColor);
       compareButtonDialog.addActionListener(this); 
       compareButtonDialog.setPreferredSize(new Dimension(250,30));
       compareButtonDialog.setMinimumSize(new Dimension(250,30));
       compareButtonDialog.setMaximumSize(new Dimension(300,30));
       executecompareHButton.addActionListener(this);
       comparepanel.add(compareButtonDialog);
       comparepanel.add(executecompareHButton);
       cp.add(comparepanel);
    }

    //////////////////////////////////////////////////////////////////////////////////
    /**
     * Writes the default settings to a file
     */
    public void printDefaults(PrintWriter pwsave) 
    {

       pwsave.println("#Main Input:");         
       pwsave.println("Data_File"+"\t"+theDataSet.szInputFile);
       pwsave.println("Gene_Annotation_Source"+"\t"+theDataSet.tga.szorganismsourceval);
       pwsave.println("Gene_Annotation_File"+"\t"+theDataSet.tga.szGoFile);
       pwsave.println("Cross_Reference_Source"+"\t"+theDataSet.tga.szxrefsourceval);
       pwsave.println("Cross_Reference_File"+"\t"+theDataSet.tga.szxrefval);
       pwsave.println("Gene_Location_Source"+"\t"+cf.genomeParser.szchromsourceval);
       pwsave.println("Gene_Location_File"+"\t"+cf.genomeParser.szchromval);

       if (theDataSet.bkmeans)
       {
	  pwsave.println("Clustering_Method[STEM Clustering Method,K-means]"+"\tK-means"); 
          pwsave.println("Number_of_Clusters_K"+"\t"+theDataSet.nmaxprofiles);
          pwsave.println("Number_of_Random_Starts"+"\t"+theDataSet.nmaxchange);      
       }
       else
       {
	  pwsave.println("Clustering_Method[STEM Clustering Method,K-means]"+"\tSTEM Clustering Method");
          pwsave.println("Maximum_Number_of_Model_Profiles"+"\t"+theDataSet.nmaxprofiles);
          pwsave.println("Maximum_Unit_Change_in_Model_Profiles_between_Time_Points"+"\t"+theDataSet.nmaxchange);
       }

       pwsave.print("Normalize_Data[Log normalize data,Normalize data,No normalization/add 0]"+"\t");
       if (theDataSet.btakelog)
       {
	   pwsave.println("Log normalize data");
       }
       else if (theDataSet.badd0)
       {
	   pwsave.println("No normalization/add 0");
       }
       else
       {
	   pwsave.println("Normalize data");
       }
       pwsave.println("Spot_IDs_included_in_the_data_file"+"\t"+theDataSet.bspotincluded);
       pwsave.println();
     
       //----------------------------------------------------------------------------------

       pwsave.println("#Repeat data");
       pwsave.print("Repeat_Data_Files(comma delimited list)"+"\t");   
       if (theDataSet.otherInputFiles != null)
       {   
          for (int nindex =0; nindex < theDataSet.otherInputFiles.length; nindex++)
          {
	     pwsave.print(theDataSet.otherInputFiles[nindex]);
             if (nindex < theDataSet.otherInputFiles.length-1)
             {
                pwsave.print(",");
             }
          }
       }
       pwsave.println();

       pwsave.print("Repeat_Data_is_from[Different time periods,The same time period]"+"\t");
       if (theDataSet.bfullrepeat)
       {
          pwsave.println("Different time periods");
       }
       else
       {
          pwsave.println("The same time period");
       }

       pwsave.println();

       //----------------------------------------------------------------------------------

       pwsave.println("#Comparison Data:");
       if (thecomparegui != null)
       {
	   pwsave.println("Comparison_Data_File"+"\t"+thecomparegui.theCompareInfo.comparesetfmnel.szInputFile);
          pwsave.print("Comparison_Repeat_Data_Files(comma delimited list)"+"\t");
	  if (theDataSet.otherInputFiles != null)
	  {
             for (int nindex =0; nindex < theDataSet.otherInputFiles.length; nindex++)
             {
	        pwsave.print(thecomparegui.theCompareInfo.comparesetfmnel.otherInputFiles[nindex]);
                if (nindex < thecomparegui.theCompareInfo.comparesetfmnel.otherInputFiles.length-1)
                {
                   pwsave.print(",");
                }
	     }
	  }
	  pwsave.println();

          pwsave.print("Comparison_Repeat_Data_is_from[Different time periods,The same time period]"+"\t");
          if (thecomparegui.theCompareInfo.comparesetfmnel.bfullrepeat)
          {
             pwsave.println("Different time periods");
          }
          else
          {
             pwsave.println("The same time period");
          }

          pwsave.println("Comparison_Minimum_Number_of_genes_in_intersection"+"\t"+
                          +thecomparegui.theCompareInfo.nminnumgenes);
          pwsave.println("Comparison_Maximum_Uncorrected_Intersection_pvalue"+"\t"
                          +thecomparegui.theCompareInfo.dmaxpval);
       }
       else
       {
          pwsave.println("Comparison_Data_File"+"\t"+ST.szCompareDEF);
          pwsave.print("Comparison_Repeat_Data_Files(comma delimited list)"+"\t");
	  if (theDataSet.otherInputFiles != null)
	  {
	     int nsize = ST.vRepeatCompareDEF.size();
             for (int nindex =0; nindex < nsize; nindex++)
             {
		pwsave.print(ST.vRepeatCompareDEF.get(nindex));
                if (nindex < nsize-1)
                {
                   pwsave.print(",");
                }
	     }
	  }
	  pwsave.println();

          pwsave.print("Comparison_Repeat_Data_is_from[Different time periods,The same time period]"+"\t");
          if (ST.bcomparealltimeDEF)
          {
             pwsave.println("Different time periods");
          }
          else
          {
             pwsave.println("The same time period");
          }

          pwsave.println("Comparison_Minimum_Number_of_genes_in_intersection"+"\t"+
                          +ST.nCompareMinGenesDEF);
          pwsave.println("Comparison_Maximum_Uncorrected_Intersection_pvalue"+"\t"
                          +ST.dCompareMinpvalDEF);
       }
       pwsave.println();

       //----------------------------------------------------------------------------------

       pwsave.println("#Filtering:");
       pwsave.println("Maximum_Number_of_Missing_Values"+"\t"+theDataSet.nmaxmissing);
       pwsave.println("Minimum_Correlation_between_Repeats"+"\t"+theDataSet.dmincorrelation);
       pwsave.println("Minimum_Absolute_Log_Ratio_Expression"+"\t"+theDataSet.dthresholdvalue);
       pwsave.print("Change_should_be_based_on[Maximum-Minimum,Difference From 0]"+"\t");
       if (theDataSet.bmaxminval)
       {
	   pwsave.println("Maximum-Minimum");
       }
       else
       {
	   pwsave.println("Difference From 0");
       }
       pwsave.println("Pre-filtered_Gene_File"+"\t"+theDataSet.tga.szextraval);
       pwsave.println();

       //----------------------------------------------------------------------------------

       if (!theDataSet.bkmeans)
       {
          pwsave.println("#Model Profiles");
          pwsave.println("Maximum_Correlation"+"\t"+theDataSet.dmaxcorrmodel);
          pwsave.println("Number_of_Permutations_per_Gene"+"\t"+theDataSet.nsamplesgene);
          pwsave.println("Maximum_Number_of_Candidate_Model_Profiles"+"\t"+theDataSet.nsamplesmodel);
          pwsave.println("Significance_Level"+"\t"+theDataSet.alpha);
          pwsave.print("Correction_Method[Bonferroni,False Discovery Rate,None]"+"\t");
          if (theDataSet.nfdr==0)
          {
	     pwsave.println("None");
          } 
          else if (theDataSet.nfdr==1)
          {
             pwsave.println("False Discovery Rate");
          }
          else if (theDataSet.nfdr==2)
          {
             pwsave.println("Bonferroni");
          }
          pwsave.println("Permutation_Test_Should_Permute_Time_Point_0"+"\t"+theDataSet.ballpermuteval);
          pwsave.println();


       //----------------------------------------------------------------------------------

         pwsave.println("#Clustering Profiles:");
         pwsave.println("Clustering_Minimum_Correlation"+"\t"+theDataSet.dminclustdist);
         pwsave.println("Clustering_Minimum_Correlation_Percentile"+"\t"+theDataSet.dpercentileclust);
         pwsave.println();
       }

       //----------------------------------------------------------------------------------

       pwsave.println("#Gene Annotations:");
       pwsave.println("Category_ID_File"+"\t"+theDataSet.tga.szcategoryIDval);    
       pwsave.println("Include_Biological_Process"+"\t"+theDataSet.tga.bpontoval);
       pwsave.println("Include_Molecular_Function"+"\t"+theDataSet.tga.bfontoval);
       pwsave.println("Include_Cellular_Process"+"\t"+theDataSet.tga.bcontoval);
       pwsave.println("Only_include_annotations_with_these_evidence_codes"+"\t"+theDataSet.tga.szevidenceval);
       pwsave.println("Only_include_annotations_with_these_taxon_IDs"+"\t"+theDataSet.tga.sztaxonval);
       pwsave.println();

       //----------------------------------------------------------------------------------
       pwsave.println("#GO Analysis:");
       pwsave.print("Multiple_hypothesis_correction_method_enrichment[Bonferroni,Randomization]"+"\t");
       if (theDataSet.tga.brandomgoval)
       {
          pwsave.println("Randomization");
       }
       else
       {
          pwsave.println("Bonferroni");
       }
       pwsave.println("Minimum_GO_level"+"\t"+theDataSet.tga.nmingolevel);
       pwsave.println("GO_Minimum_number_of_genes"+"\t"+theDataSet.tga.nmingo);
       pwsave.println("Number_of_samples_for_randomized_multiple_hypothesis_correction"+"\t"+theDataSet.tga.nsamplespval);
       pwsave.println();

       //----------------------------------------------------------------------------------

       pwsave.println("#Interface Options");
       pwsave.print(
         "Gene_display_policy_on_main_interface[Do not display,Display only selected,Display all]"+"\t");

       if (ngenedisplay == 0)
       {
	   pwsave.println("Do not display");
       }
       else if (ngenedisplay == 1)
       {
           pwsave.println("Display only selected");
       }
       else
       {
	   pwsave.println("Display all");
       }
       pwsave.println("Gene_Color(R,G,B)"+"\t"+thegenecolor.getRed()+","
		   +thegenecolor.getGreen()+","+thegenecolor.getBlue());

       try
       {
          thegeneplotpanel.thespinnerminy.commitEdit();
          thegeneplotpanel.thespinnermaxy.commitEdit();
          thegeneplotpanel.thespinnertick.commitEdit();
          thegeneplotpanel.thespinnertickmain.commitEdit();
       }
       catch(ParseException pex)
       {
          System.out.println("Warning could not correctly parse y-axis parameters");
       }

       double dminfixed = ((Double) thegeneplotpanel.thespinnerminy.getValue()).doubleValue();
       double dmaxfixed = ((Double) thegeneplotpanel.thespinnermaxy.getValue()).doubleValue();
       double dtickinterval = ((Double) thegeneplotpanel.thespinnertick.getValue()).doubleValue();
       double dtickintervalmain = ((Double) thegeneplotpanel.thespinnertickmain.getValue()).doubleValue();

       if (theDataSet.bkmeans)
       {
	  pwsave.println("Display_Cluster_Mean\t"+bdisplayprofileline);  
          pwsave.println("Display_K-Means_Cluster_ID\t"+bdisplayID);
       }
       else
       {
          pwsave.println("Display_Model_Profile\t"+bdisplayprofileline);   
	  pwsave.println("Display_Profile_ID\t"+bdisplayID);
       }
       pwsave.println("Display_details_when_ordering\t"+bdisplaydetail);
       pwsave.println("Show_Main_Y-axis_gene_tick_marks\t"+bdisplaymaintick);
       pwsave.println("Main_Y-axis_gene_tick_interval\t"+dtickintervalmain);

       if (theDataSet.bkmeans)
       {
          pwsave.print("Y-axis_scale_for_genes_on_k-means_main_interface_should_be[Cluster specific,Global]"+"\t");
       }
       else
       {
          pwsave.print(
              "Y-axis_scale_for_genes_on_main_interface_should_be[Gene specific,Profile specific,Global]"+"\t");
       }
       if (ngenescale ==0)
       {
          pwsave.println("Gene specific");
       }
       else if (ngenescale == 1)
       {
          pwsave.println("Profile specific");
       }
       else
       {
	  pwsave.println("Global");
       }

       pwsave.println("Scale_should_be_based_on_only_selected_genes"+"\t"+bscalevisible);

       boolean bautomatic = thegeneplotpanel.automaticButton.isSelected();

       pwsave.print("Y-axis_scale_on_details_windows_should_be[Determined automatically,Fixed]"+"\t");  
       if (bautomatic)
       {
	  pwsave.println("Determined automatically");
       }
       else
       {
          pwsave.println("Fixed");
       }

       pwsave.println("Y_Scale_Min"+"\t"+dminfixed);
       pwsave.println("Y_Scale_Max"+"\t"+dmaxfixed);
       pwsave.println("Tick_interval"+"\t"+dtickinterval);

       pwsave.print("X-axis_scale_should_be[Uniform,Based on real time]"+"\t");	
       if (buniformXaxis)
       {
	  pwsave.println("Uniform");
       }
       else 
       {
	  pwsave.println("Based on real time");
       }
       	        
       pwsave.close();
    }
}

