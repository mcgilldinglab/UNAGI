package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.chromviewer.*;
import edu.cmu.cs.sb.core.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.awt.image.*;
import javax.swing.*;
import java.util.*;
import java.text.NumberFormat;
import java.io.*;
import java.text.*;


/**
 * Class for interface display for all the genes assigned to the same profile 
 */
public class ProfileGui extends JFrame implements MouseListener, ComponentListener
{

    Object sizeLock = new Object();
    int nnewheight = -1;
    int nnewwidth = -1;
    ChromFrame cf;
    String szprofilecluster;
    String szprofileclusterCAP;

    Dimension totalSize;
    Graphics2D g2;
     
    STEM_DataSet theDataSet;
    int nprofile;
    int ncluster;
    String szAssignInfo;
    String szGene, szSpot;
    GoAnnotations.ProfileGORankingRec pgrr;
    boolean bshowcluster;
    HashSet inames;
    int ncompareprofile;
    String szIntersect;
    boolean bonlyorder;
    int noffset;
    boolean bcheckonly;
    boolean bgosorttable;
    int[] ranklookup;
    int nmidpoint;
    GoAnnotations.Rec gar;
    boolean bautomatic;
    double dminfixed;
    double dmaxfixed;
    double dtickinterval;
    GenePlotPanel thegeneplotpanel;

    int SCREENWIDTH = MAINGUI2.SCREENWIDTH;
    int SCREENHEIGHT = MAINGUI2.SCREENHEIGHT;
    static int SPACETOP = 70;
    static int SPACEBOT2 =20;
    static int SPACEBOT = 50;
    static int SPACELEFT = 40;
    static int SPACERIGHT = 40;
    //800-80=720/2=360+40

    int ncolwidth;
    int nrowheight;
    int numcols;
    int numrows;
    int profiletablex;
    int clustertablex;
    int querytabley;
    int tabley;
    int tablewid;
    int tableheight;
    int nonlyx;
    int nonlyy;
    int ngosorttabley;
    int nonlytablewid;
    int ngosorttablewid;
    int ngosorttablex;
    int profilegox;
    int clustergox;

    ArrayList colorlist = new ArrayList();

    final static int maxCharHeight = 15;
    final static int minFontSize = 6;
    final static int BSPACING = 15;
    final static Color bg = Color.white;
    final static Color fg = Color.black;
    final static Color red = Color.red;
    final static Color white = Color.white;
    final static BasicStroke stroke = new BasicStroke(2.0f);
    final static BasicStroke wideStroke = new BasicStroke(8.0f);
    final static float dash1[] = {10.0f};
    final static BasicStroke dashed = new BasicStroke(1.0f, 
                                                      BasicStroke.CAP_BUTT, 
                                                      BasicStroke.JOIN_MITER, 
                                                      10.0f, dash1, 0.0f);

    /**
     * Constructor renders the display interface
     */
    public ProfileGui(STEM_DataSet theDataSet, int nprofile,int[] ranklookup,
		      HashSet inames, int ncompareprofile,String szIntersect,String szGene, String szSpot,
                      GenePlotPanel thegeneplotpanel, ChromFrame cf)
    {
        super((theDataSet.bkmeans? "Cluster" : "Profile")+" "+nprofile);

	this.thegeneplotpanel = thegeneplotpanel;
   
	if (theDataSet.bkmeans)
	{
	    szprofilecluster = "cluster";
	    szprofileclusterCAP = "Cluster";
	}
	else
	{
	    szprofilecluster = "profile";
	    szprofileclusterCAP = "Profile";
	}

	addComponentListener(this);
        addMouseListener(this);

        this.theDataSet = theDataSet;
        this.nprofile = nprofile;
        this.inames = inames;
        this.ncompareprofile = ncompareprofile;
	this.szIntersect = szIntersect;
	this.bonlyorder = false;
        this.ranklookup = ranklookup;
        this.szGene = szGene;
        this.szSpot = szSpot;
	this.cf = cf;

        ncluster = -1;
        int nindex = 0;
        boolean bfound = false;
        while ((nindex < theDataSet.clustersofprofilesnum.size()) &&(!bfound))
        {
           ArrayList currcluster = (ArrayList) theDataSet.clustersofprofilesnum.get(nindex);
           int njindex = 0;
           while ((njindex < currcluster.size())&&(!bfound))
           {
              STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) currcluster.get(njindex);
              if (pr.nprofileindex == nprofile)
              {
                 ncluster = nindex;
                 bfound = true;
              }
              njindex++;
           }
           nindex++;
        }

        if ((ncluster >=0) && (((ArrayList) theDataSet.clustersofprofilesnum.get(ncluster)).size()>1))
        {
	    bshowcluster = true;
	}
        else
	{
            bshowcluster =false;
	}
	init();
    }

    /**
     * Empty method
     */
    public void componentHidden(ComponentEvent e) 
    {
    }

    /**
     * Empty method
     */
    public void componentMoved(ComponentEvent e) 
    {
    }

    /**
     * Empty method
     */
    public void componentResized(ComponentEvent e)  
    {
       synchronized (sizeLock)
       {
          int nheight = getHeight();
	  int nwidth = getWidth();

	  if (((nheight != SCREENHEIGHT) || (nwidth != SCREENWIDTH))&&
	     ((nheight != nnewheight) || (nwidth != nnewwidth)))
	  {
	     
	     nnewheight = nheight;
	     nnewwidth = nwidth;
	     repaint();
	  }
       }
    }

    /**
     * Empty method
     */
    public void componentShown(ComponentEvent e)  
    {
    }

    /**
     * Converts the number of genes enriched as a double to a string representation
     */
    public String enrichedToSz(double dexpected)
    {
           NumberFormat nf1 =  NumberFormat.getInstance(Locale.ENGLISH);
           nf1.setMinimumFractionDigits(1);
           nf1.setMaximumFractionDigits(1);
           nf1.setGroupingUsed(false);

           String sz;
           sz = nf1.format(dexpected);
           double dval = Double.parseDouble(sz);
           if (dval > 0)
           {
              sz = "+" + sz;
           }
           else if (dval == 0)
	   {
	      sz = "0.0";
	   }
	      
           return sz;
    }


    /**
     * Initializes the colors
     */
    public void init() 
    {
        setBackground(bg);
        setForeground(fg);
    }

    /**
     * Empty constructor method
     */
    public void mousePressed(MouseEvent e) 
    {
    }

    /**
     * Empty constructor method
     */
    public void mouseReleased(MouseEvent e)
    {
    }

    /**
     * Empty constructor method
     */
    public void mouseEntered(MouseEvent e) 
    {
    }

    /**
     * Empty constructor method
     */
    public void mouseExited(MouseEvent e) 
    {
    }


    /**
     * Draws button for profile or k-means gene table
     */
    void drawprofiletable(Color fillColor)
    {
       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(profiletablex,tabley,tablewid,tableheight));
       g2.setColor(Color.black);
       g2.draw(new Rectangle2D.Double(profiletablex,tabley,tablewid,tableheight));

       if(theDataSet.bkmeans)
       {
           g2.drawString("Cluster Gene Table", profiletablex+24,tabley+15);
       }
       else
       {
           g2.drawString("Profile Gene Table", profiletablex+27,tabley+15);
       }
    }

    /**
     * Draws button for a cluster of profiles gene table
     */
    void drawclustertable(Color fillColor)
    {
       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(clustertablex,tabley,tablewid,tableheight));
       g2.setColor(Color.black);
       g2.draw(new Rectangle2D.Double(clustertablex,tabley,tablewid,tableheight));
       g2.drawString("Cluster Gene Table", clustertablex+24,tabley+15);
    }

    /**
     * Draws button for profile or k-means GO table
     */
    void drawprofilego(Color fillColor)
    {
       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(profilegox,tabley,tablewid,tableheight));
       g2.setColor(Color.black);
       if(theDataSet.bkmeans)
       {
          g2.drawString("Cluster GO Table", profilegox+30,tabley+15);
       }
       else
       {
          g2.drawString("Profile GO Table", profilegox+33,tabley+15);
       }

       g2.draw(new Rectangle2D.Double(profilegox,tabley,tablewid,tableheight));
    }

    /**
     * Draws button to give option to display only a subset of genes
     */
    void drawonlyorder()
    {
       g2.setColor(ST.buttonColor);
       FontMetrics fm = g2.getFontMetrics();
       Rectangle2D bounds = fm.getStringBounds("Click to plot all "+szprofilecluster+" genes",g2);

       int nclickminwid = (int) Math.ceil(bounds.getMaxX());
       String szonlylabel= "Click to plot ";
       if (bonlyorder)
       {
	   szonlylabel += "all "+szprofilecluster+" genes";
       }
       else if (inames != null)
       {
          szonlylabel += "only genes in intersection";
       }
       else if (szSpot != null)
       {
           szonlylabel += "only gene "+szGene;
       }
       else if ((pgrr != null)&&((theDataSet.tga.szsortcommand.equals("expgo"))||
                                  (theDataSet.tga.szsortcommand.equals("go"))))
       {
	  gar = (GoAnnotations.Rec) theDataSet.tga.htGO.get(theDataSet.tga.szSelectedGO);
	  szonlylabel +="only "+szprofilecluster+" "+gar.sztermName+" genes";
       }
       else
       {
	  szonlylabel +="only "+szprofilecluster+" query set genes";
       }

       int nextraspace = 15;
       int nextraspaceleft;

       if (!bonlyorder)
       {      
          fm = g2.getFontMetrics();
          bounds = fm.getStringBounds(szonlylabel,g2);
	  nonlytablewid = Math.max((int) Math.ceil(bounds.getMaxX()),nclickminwid)+nextraspace;
      
          nonlyx = nmidpoint-nonlytablewid/2;
          nextraspaceleft = (int) Math.ceil(nextraspace/2.0);
       }
       else
       {
	   nextraspaceleft = (int) Math.ceil((nonlytablewid-nclickminwid)/2.0);
       }

       g2.fill(new Rectangle2D.Double(nonlyx,nonlyy,nonlytablewid,tableheight));
       g2.fill(new Rectangle2D.Double(nonlyx,nonlyy,nonlytablewid,tableheight));
       g2.setColor(Color.black);
       g2.drawString(szonlylabel, nonlyx+nextraspaceleft,nonlyy+nextraspace);
       g2.draw(new Rectangle2D.Double(nonlyx,nonlyy,nonlytablewid,tableheight));
    }

    /////////////////////////////////////////////////////////////////
    /**
     * Draws button to give option to display gene table for a specified profile GO subset
     */
    void drawgosorttable()
    {
       g2.setColor(ST.buttonColor);

       GoAnnotations.Rec gar = (GoAnnotations.Rec) 
                                  theDataSet.tga.htGO.get(theDataSet.tga.szSelectedGO);

       String szgosorttablelabel = szprofileclusterCAP+" "+gar.sztermName+" Gene Table";
       
       FontMetrics fm = g2.getFontMetrics();
       Rectangle2D bounds = fm.getStringBounds(szgosorttablelabel,g2);
      
       ngosorttablewid = (int) Math.ceil(bounds.getMaxX())+15;
       ngosorttablex =  nmidpoint-ngosorttablewid/2;
 

       g2.fill(new Rectangle2D.Double(ngosorttablex,ngosorttabley,ngosorttablewid,tableheight));
       g2.fill(new Rectangle2D.Double(ngosorttablex,ngosorttabley,ngosorttablewid,tableheight));
       g2.setColor(Color.black);
       g2.drawString(szgosorttablelabel, ngosorttablex+8,ngosorttabley+15);
       g2.draw(new Rectangle2D.Double(ngosorttablex,ngosorttabley,ngosorttablewid,tableheight));
    }
    
    /**
     * Draws button to give option to display gene table for a specified cluster GO subset
     */
    void drawclustergo(Color fillColor)
    {
	g2.setColor(fillColor);
        g2.fill(new Rectangle2D.Double(clustergox,tabley,tablewid,tableheight));
        g2.setColor(Color.black);
        g2.draw(new Rectangle2D.Double(clustergox,tabley,tablewid,tableheight));
        g2.drawString("Cluster GO Table", clustergox+30,tabley+15);
    }

    /**
     * Draws button to display gene table for genes in the query set that are also in the profile or k-means cluster
     */
    void drawprofilequerytable(Color fillColor)
    {
       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(profiletablex,querytabley,tablewid,tableheight));
       g2.setColor(Color.black);
       g2.draw(new Rectangle2D.Double(profiletablex,querytabley,tablewid,tableheight));

       if (theDataSet.bkmeans)
       {
          g2.drawString("Cluster Query Gene Table", profiletablex+7,querytabley+15);
       }
       else
       {
          g2.drawString("Profile Query Gene Table", profiletablex+9,querytabley+15);
       }
    }

    /**
     * Draws button for a gene table for set of genes in the gene set query assigned to the cluster of profiles
     */
    void drawclusterquerytable(Color fillColor)
    {
       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(clustertablex,querytabley,tablewid,tableheight));
       g2.setColor(Color.black);
       g2.draw(new Rectangle2D.Double(clustertablex,querytabley,tablewid,tableheight));
       g2.drawString("Cluster Query Gene Table", clustertablex+7,querytabley+15);
    }

    /**
     * Draws button for a GO table for set of genes in the gene set query assigned to the same profile or k-means cluster
     */
    void drawprofilequerygo(Color fillColor)
    {
       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(profilegox,querytabley,tablewid,tableheight));
       g2.setColor(Color.black);

       if (theDataSet.bkmeans)
       {
          g2.drawString("Cluster Query GO Table", profilegox+12,querytabley+15);
       }
       else
       {
          g2.drawString("Profile Query GO Table", profilegox+15,querytabley+15);
       }
       g2.draw(new Rectangle2D.Double(profilegox,querytabley,tablewid,tableheight));
    }

    /**
     * Draws button for a GO table for set of genes in the gene set query assigned to the cluster of profiles
     */
    void drawclusterquerygo(Color fillColor)
    {
	g2.setColor(fillColor);
        g2.fill(new Rectangle2D.Double(clustergox,querytabley,tablewid,tableheight));
        g2.setColor(Color.black);
        g2.draw(new Rectangle2D.Double(clustergox,querytabley,tablewid,tableheight));
        g2.drawString("Cluster Query GO Table", clustergox+12,querytabley+15);
    }

    /**
     * Draws button for gene table for set of genes in a comparison intersection
     */
    void drawintersecttable(Color fillColor)
    {

       g2.setColor(fillColor);
       g2.fill(new Rectangle2D.Double(profiletablex,querytabley,tablewid,tableheight));
       g2.setColor(Color.black);
       g2.draw(new Rectangle2D.Double(profiletablex,querytabley,tablewid,tableheight));

       if (theDataSet.bkmeans)
       {
	   g2.drawString("Cluster Intersect Gene Table", profiletablex,querytabley+15);
       }
       else
       {
          g2.drawString("Profile Intersect Gene Table", profiletablex+2,querytabley+15);
       }
    }

    /**
     * Draws button for GO table for set of genes in a comparison intersection
     */
    public void drawintersectgo(Color fillColor)
    {
	g2.setColor(fillColor);
        g2.fill(new Rectangle2D.Double(profilegox,querytabley,tablewid,tableheight));
        g2.setColor(Color.black);
        g2.draw(new Rectangle2D.Double(profilegox,querytabley,tablewid,tableheight));
        g2.drawString(szprofileclusterCAP+" Intersect GO Table", profilegox+8,querytabley+15);
    }


    /**
     * Responds to mouse clicked events
     */ 
    public void mouseClicked(MouseEvent e) 
    {
       if (e.getButton() == MouseEvent.BUTTON1)
       {
	   int nypoint = e.getY();
	   int nxpoint = e.getX();
	   if ((bgosorttable)&&(ngosorttabley <= nypoint)&&(nypoint <= ngosorttabley+tableheight)&&
                 (ngosorttablex<= nxpoint)&&(nxpoint <= ngosorttablex+ngosorttablewid))
	   {
               javax.swing.SwingUtilities.invokeLater(new Runnable() 
               {
                  public void run() 
                  {
                     //id then title
	             gar = (GoAnnotations.Rec) 
                                  theDataSet.tga.htGO.get(theDataSet.tga.szSelectedGO);
                     String szGoCombined = theDataSet.tga.szSelectedGO +
			           " ("+gar.sztermName+")";
                     String sznewTitle = "Gene List for "+szGoCombined +
                                          " genes in "+ szprofileclusterCAP+" " + nprofile;

                     JFrame frame = new JFrame(sznewTitle);
                     frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
                        
                     String szHeaderInfo = "";                  

                     GOGeneTable newContentPane = new GOGeneTable(frame,theDataSet, 
							 theDataSet.tga.szSelectedGO,
                                                                     szGoCombined,
                                                                     nprofile, 
                                                                     szHeaderInfo,
								     null,
								 false,null,sznewTitle,false,cf);
                     newContentPane.setOpaque(true); //content panes must be opaque
                     frame.setContentPane(newContentPane);
                     frame.setLocation(40,150);
                     //Display the window.
                     frame.pack();
                     frame.setVisible(true);
	           }
	      });
	   }
 	   else if ((bcheckonly)&&(nonlyy <= nypoint)&&(nypoint <= nonlyy+tableheight)&&
                 (nonlyx<= nxpoint)&&(nxpoint <= nonlyx+nonlytablewid))
	   {
	      bonlyorder = !bonlyorder;
	      repaint();
	     	       
	   } 
           else if ((tabley <= nypoint)&&(nypoint <= tabley+tableheight))
	   {
              if ((profiletablex<= nxpoint)&&
                 (nxpoint <= profiletablex+tablewid))
	      {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                    public void run() 
                    {
                       drawprofiletable(Color.gray);
                       String szTitle ="Gene Table for "+szprofileclusterCAP+" "+nprofile;
                       JFrame frame = new JFrame(szTitle);
                       frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	               frame.setLocation(25,100);
                       GeneTable newContentPane = new GeneTable(frame,theDataSet,nprofile,
							      szAssignInfo,false,null,szTitle,cf);
                       newContentPane.setOpaque(true);
                       frame.setContentPane(newContentPane);
                       drawprofiletable(ST.buttonColor);
                       //Display the window.
                       frame.pack();
                       frame.setVisible(true);
		    }
		});	       
	      }
              else if  ((profilegox<= nxpoint)&& (nxpoint <= profilegox+tablewid))
	      {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                    public void run() 
                    {
                       drawprofilego(Color.gray);
	               String szTitle ="GO Results for "+szprofileclusterCAP+" "+nprofile+
                                  " based on the actual number of genes assigned to the "+szprofilecluster;

                       JFrame frame = new JFrame(szTitle);
                       frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		       frame.setLocation(25,100);
	 	       GOTable newContentPane = new GOTable(frame,theDataSet, nprofile,null,false,null,szTitle,false,cf);
                       drawprofilego(ST.buttonColor);
                       newContentPane.setOpaque(true); //content panes must be opaque
                       frame.setContentPane(newContentPane);
                       //Display the window.
                      frame.pack();
                      frame.setVisible(true);	 
		    }
	         });
	      }
              else if ((bshowcluster) && (clustergox <= nxpoint)&&
		       (nxpoint <= clustergox+tablewid))
	      {
                 javax.swing.SwingUtilities.invokeLater(new Runnable() 
                 {
                  public void run() 
                  {
                     drawclustergo(Color.gray);   
		     ArrayList alcprofiles =  (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
		     String szcprofiles = clusterArrayToString(alcprofiles);
	             String szTitle ="GO Results for Cluster "+ncluster +" "+szcprofiles
                                    +" based on the actual number of genes assigned to the cluster"; 
                     JFrame frame = new JFrame(szTitle);
                     frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		     frame.setLocation(25,100);
 
  		     GOTable newContentPane = new GOTable(frame,theDataSet, ncluster,
						    alcprofiles,
							  false,null,szTitle,false,cf);
                     newContentPane.setOpaque(true); //content panes must be opaque
                     frame.setContentPane(newContentPane);
                     drawclustergo(ST.buttonColor);
     
                     //Display the window.
                     frame.pack();
                     frame.setVisible(true);	 
		  }
	        });
	     }
             else if ((bshowcluster) && (clustertablex <= nxpoint)&&
		   (nxpoint <= clustertablex+tablewid))
	     {
	    
                javax.swing.SwingUtilities.invokeLater(new Runnable() 
                {
                   public void run() 
                   {
                      drawclustertable(Color.gray);   
		      ArrayList alcprofiles = (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
		      String szcprofiles = clusterArrayToString(alcprofiles);
	              String szTitle = "Gene Table for Cluster "+ncluster + " "+szcprofiles;

                      JFrame frame = new JFrame(szTitle);
                      frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	              frame.setLocation(25,100);
                      szAssignInfo = ""+ncluster;
                      GeneTable newContentPane = new GeneTable(frame,theDataSet, alcprofiles,
							       szAssignInfo,false,null,szTitle,cf);
                      newContentPane.setOpaque(true); //content panes must be opaque
                      frame.setContentPane(newContentPane);
 
                      drawclustertable(ST.buttonColor);
     
                      //Display the window.
                      frame.pack();
                      frame.setVisible(true);
		   }
		});	 
	     }
	  }
          else if ((querytabley <= nypoint)&&(nypoint <= querytabley+tableheight)&&
                   ((inames != null)||
		    ((pgrr != null)&& ((theDataSet.tga.szsortcommand.equals("define"))||
				       (theDataSet.tga.szsortcommand.equals("expdefine"))))))
	  {
             if ((profiletablex<= nxpoint)&&
                 (nxpoint <= profiletablex+tablewid))
	     {
                javax.swing.SwingUtilities.invokeLater(new Runnable() 
                {
                   public void run() 
                   {
		      JFrame frame;
                      if (inames != null)
		      {
                         drawintersecttable(Color.gray);
                         String szTitle ="Gene Table for Intersection of (row) "+szprofileclusterCAP+
                                    " "+ncompareprofile
		                    +" with (column) "+szprofileclusterCAP+" "+nprofile; 
                         frame = new JFrame(szTitle);
                         frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	                 frame.setLocation(25,100);
                         GeneTable newContentPane = new GeneTable(frame,theDataSet,nprofile,
								  szAssignInfo,false,inames,szTitle,cf);
                         newContentPane.setOpaque(true); //content panes must be opaque
                         frame.setContentPane(newContentPane);
                         drawintersecttable(ST.buttonColor);
		      }
  		      else
       		      {
                         drawprofilequerytable(Color.gray);
	    	         String szTitle ="Gene Table for Query Genes with "+szprofileclusterCAP+" "+nprofile;
                         frame = new JFrame(szTitle);
                         frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	                 frame.setLocation(25,100);
                         GeneTable newContentPane = new GeneTable(frame,theDataSet,nprofile,
								  szAssignInfo,true,null,szTitle,cf);
                         newContentPane.setOpaque(true); //content panes must be opaque
                         frame.setContentPane(newContentPane);
                         drawprofilequerytable(ST.buttonColor);
                      }
                      //Display the window.
                      frame.pack();
                      frame.setVisible(true);	       
		   }
		});
	    } 
            else if  ((profilegox<= nxpoint)&&
                      (nxpoint <= profilegox+tablewid))
	    {
               javax.swing.SwingUtilities.invokeLater(new Runnable() 
               {
                  public void run() 
                  {
                     JFrame frame;    
                     GOTable newContentPane;            
                     if (inames != null)
	             {
                        drawintersectgo(Color.gray);
                        String szTitle = "GO Results for Intersection of (row) "+szprofileclusterCAP+" "
                        +ncompareprofile+" with (column) "+szprofileclusterCAP+" "+nprofile+
		        " based on the actual number of genes in the intersection";
                        frame = new JFrame(szTitle);
                        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	                frame.setLocation(25,100);
		        newContentPane = new GOTable(frame,theDataSet, nprofile,null,false,inames,szTitle,false,cf);
                        drawintersectgo(ST.buttonColor);
	             }
                     else
	             { 
                        drawprofilequerygo(Color.gray);
		        String szTitle = "GO Results for Query Genes with "+szprofileclusterCAP+" "+nprofile+
                                         " based on the actual number of genes in the set";
                        frame = new JFrame(szTitle);
                        frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	                frame.setLocation(25,100);
		        newContentPane = new GOTable(frame,theDataSet, nprofile,null,true,null,szTitle,false,cf);
                        drawprofilequerygo(ST.buttonColor);
	             }

                     newContentPane.setOpaque(true); //content panes must be opaque
                     frame.setContentPane(newContentPane);
       
                    //Display the window.
                    frame.pack();
                    frame.setVisible(true);	 
		  }
	        });    
	     }
             else if ((bshowcluster) && (clustergox <= nxpoint)&&
		   (nxpoint <= clustergox+tablewid))
	     {
                javax.swing.SwingUtilities.invokeLater(new Runnable() 
                {
                   public void run() 
                   {
                      drawclusterquerygo(Color.gray);   
		      ArrayList alcprofiles =  (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
		      String szcprofiles = clusterArrayToString(alcprofiles);
	              String szTitle = "GO Results for Query Genes in Cluster "+ncluster+" "+szcprofiles+
                                        " based on the actual number of genes in the cluster";
                      JFrame frame = new JFrame(szTitle);
                      frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
		      frame.setLocation(25,100);
 
	    	      GOTable newContentPane = new GOTable(frame,theDataSet, ncluster,
							   alcprofiles,true,null,szTitle,false,cf);
                      newContentPane.setOpaque(true); //content panes must be opaque
                      frame.setContentPane(newContentPane);
                      drawclusterquerygo(ST.buttonColor);
     
                      //Display the window.
                      frame.pack();
                      frame.setVisible(true);
		   }
		});	 
	     }
             else if ((bshowcluster) && (clustertablex <= nxpoint)&&
		   (nxpoint <= clustertablex+tablewid))
	     {
                javax.swing.SwingUtilities.invokeLater(new Runnable() 
                {
                   public void run() 
                   {
                      drawclusterquerytable(Color.gray);   
		      ArrayList alcprofiles =  (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
		      String szcprofiles = clusterArrayToString(alcprofiles);
	              String szTitle = "Gene Table for Query Genes in Cluster "+ncluster  + " "+szcprofiles+     
                                      " based on the actual number of genes in the cluster";
                      JFrame frame = new JFrame(szTitle);
                      frame.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
	              frame.setLocation(25,100);
                      szAssignInfo = ""+ncluster;
                      GeneTable newContentPane = new GeneTable(frame,theDataSet, 
							       alcprofiles,szAssignInfo,true,null,szTitle,cf);
                      newContentPane.setOpaque(true); //content panes must be opaque
                      frame.setContentPane(newContentPane);
 
                      drawclusterquerytable(ST.buttonColor);
     
                      //Display the window.
                      frame.pack();
                      frame.setVisible(true);	 
		   }
		});
	     }
	  }
       } 

    }


    /**
     * Static method that returns an array of profiles, a string with their profile IDs
     */
    public synchronized static String clusterArrayToString(ArrayList alcprofiles)
    {
       String sz;
       if (alcprofiles.size()==1)
       {
          STEM_DataSet.ProfileRec pr0 =  (STEM_DataSet.ProfileRec) alcprofiles.get(0);
	  sz = "(Profile "+pr0.nprofileindex+")";
       }
       else if (alcprofiles.size()==2)
       {
	  sz = "(Profiles ";
          STEM_DataSet.ProfileRec pr0 =  (STEM_DataSet.ProfileRec) alcprofiles.get(0);
	  STEM_DataSet.ProfileRec pr1 =  (STEM_DataSet.ProfileRec) alcprofiles.get(1);
	  if (pr0.nprofileindex < pr1.nprofileindex)
	  {
             sz += pr0.nprofileindex+" and "+pr1.nprofileindex+")";
	  }
	  else
	  {
             sz += pr1.nprofileindex+" and "+pr0.nprofileindex+")";
	  }
       }
       else
       {
          STEM_DataSet.ProfileRec pr;
	  int[] ncprofilesA = new int[alcprofiles.size()];
	  
	  StringBuffer szBuf = new StringBuffer("(Profiles ");
	  for (int nindex = 0; nindex < alcprofiles.size(); nindex++)
	  {
             pr =  (STEM_DataSet.ProfileRec) alcprofiles.get(nindex);
             ncprofilesA[nindex] = pr.nprofileindex;			      
	  } 
	  Arrays.sort(ncprofilesA);
	  for (int ncprofile = 0; ncprofile < ncprofilesA.length-1; ncprofile++)
	  {
	     szBuf.append(ncprofilesA[ncprofile]+", ");
	  }
          szBuf.append("and "+ncprofilesA[ncprofilesA.length-1]+")");
	  sz = szBuf.toString();
       }

       return sz;
    }


    /////////////////////////////////////////////////////////////////////////////////
    /**
     * Responsible for rendering the profile window interface
     */
    public void plotgenevalues(Graphics2D g2,double numassigned,double numexpected,double pvalue, 
                                double[][] genevalues,int[][] missing,ArrayList[] bestAssignments,
			       double[] profilevals, boolean[] sig, ArrayList currProfileAssignments)
    {
         
	g2.setColor(Color.white);
	synchronized (sizeLock)
	{
	   SCREENWIDTH = getWidth();
	   SCREENHEIGHT = getHeight();
	}

        g2.fill(new Rectangle2D.Double(0,0,SCREENWIDTH,SCREENHEIGHT));
        int numuniquegenes = currProfileAssignments.size();
        double dmaxval = 0;
	double dminval = 0;

	if (bautomatic)
	{
	   for (int nindex =0; nindex < numuniquegenes; nindex++)
	   {
              int nrow =  ((Integer) currProfileAssignments.get(nindex)).intValue();
	      if ((theDataSet.modelprofiles.length>1)||(!bonlyorder)||
		 ((inames!=null)&&(inames.contains(theDataSet.genenames[nrow])))|| 
		 ((szSpot !=null)&&(theDataSet.probenames[nrow].equals(szSpot)))||
		 ((szSpot==null)&&(inames==null)&&
                 (theDataSet.tga.isOrder(theDataSet.genenames[nrow]))))
	      {
                 //if only one model profile then always rescale otherwise hold steady
                 for (int ncol = 0; ncol < genevalues[nrow].length; ncol++)
	         {
		    if ((missing[nrow][ncol]>0)&&(Math.abs(genevalues[nrow][ncol]) > dmaxval))
		    {
		       dmaxval = Math.abs(genevalues[nrow][ncol]);
		    }
	         }
	      }
	   }

           if (dmaxval == 0)
	   {
	      dmaxval = 1;
           }
	   dminval = -dmaxval;
	}
	else
	{
	    dmaxval = dmaxfixed;
	    dminval = dminfixed;

	    if ((dminval==0)&&(dmaxval==0))
	    {
		dminval = -.1;
		dmaxval = .1;
	    }
	}

	int npower=0;
	int ndecimalpower = 0;
	double dpowscale;
	double MAXFOLD, MINFOLD;
        NumberFormat nftick = NumberFormat.getInstance(Locale.ENGLISH);

	if (bautomatic)
	{
	   if (dmaxval > 0)
	   {
	      npower=(int) Math.floor(Math.log(dmaxval)/Math.log(10));
	   }
	
   	   if (dminval < 0)
	   {
	      npower=Math.max(npower,(int) Math.floor(Math.log(-dminval)/Math.log(10)));
	   }

	   if (npower < 0)
	   {
	      nftick.setMaximumFractionDigits(-npower);
	      nftick.setMinimumFractionDigits(-npower);
	   }
	   else
	   {
	      nftick.setMaximumFractionDigits(0);
	      nftick.setMinimumFractionDigits(0);
	   }

	   dpowscale = Math.pow(10,npower);
	   MAXFOLD = (int) (Math.ceil(dmaxval/dpowscale));
	   dmaxval = MAXFOLD*dpowscale;
           MINFOLD =  (int) (Math.ceil(-dminval/dpowscale));
	   dminval = -MINFOLD*dpowscale;
	   dtickinterval = dpowscale;
	}
	else
	{
	   double dtemptickinterval = dtickinterval;
	   while (dtemptickinterval - (int) dtemptickinterval > .000001)
	   {
              ndecimalpower--;
	      dtemptickinterval=dtemptickinterval*10;
	   }

	   nftick.setMaximumFractionDigits(-ndecimalpower);
	   nftick.setMinimumFractionDigits(-ndecimalpower);
	   if ((dtickinterval - (int) dtickinterval) > 0.000001)
	   {
	      //add additional power for being greater than 0
	      npower=(int) Math.floor(Math.log(dtickinterval)/Math.log(10));
	   }
	   MAXFOLD = dmaxval;
	   MINFOLD = -dminval;
	   dpowscale = 1;
	   //powscale is always 1 for automatic
	}

        double NUMTICKS = MAXFOLD+MINFOLD;        
	double DELTAY;

	if (bautomatic)
	{
           DELTAY = (int) (SCREENHEIGHT-(SPACETOP+SPACEBOT+SPACEBOT2))/NUMTICKS;
	}
	else
	{
	   DELTAY = (SCREENHEIGHT-(SPACETOP+SPACEBOT+SPACEBOT2))/NUMTICKS;
	}
        int numpoints = genevalues[0].length;
        int DELTAX = (SCREENWIDTH-SPACELEFT-SPACERIGHT) /(numpoints-1);
        NumberFormat nf = NumberFormat.getInstance(Locale.ENGLISH);
        nf.setMinimumFractionDigits(1);
        nf.setMaximumFractionDigits(1);  
    
	int nzero = (int) ((double) SPACETOP + MAXFOLD*DELTAY);
        String szexp = Util.doubleToSz(pvalue);
	StringBuffer szTitleBuf = new StringBuffer(szprofileclusterCAP+" #"+nprofile+": (");
        for (int nindex = 0; nindex < profilevals.length-1; nindex++)
	{
	    if (theDataSet.bkmeans)
	    {
               szTitleBuf.append(nf.format(profilevals[nindex])+", ");
	    }
	    else
	    {
               szTitleBuf.append(((int) profilevals[nindex])+", ");
	    }
        }

        if (theDataSet.bkmeans)
        {
	   szTitleBuf.append(nf.format(profilevals[profilevals.length-1])+")");
	}
	else
	{
           szTitleBuf.append(((int) profilevals[profilevals.length-1])+")");
	}

	String szTitleString = szTitleBuf.toString();

        String szsig;
        if (sig[nprofile])
	{
	   szsig = " (significant)";
	}
        else
	{
           szsig = " (not significant)";
	}
	  
        g2.setColor(Color.black);

        FontMetrics fm = g2.getFontMetrics();
        Rectangle2D bounds = fm.getStringBounds(szTitleString,g2);
        int nwid = (int) Math.ceil(bounds.getMaxX());
        g2.drawString(szTitleString, SCREENWIDTH/2-nwid/2, 50);

	String szSecondLine;
	if (theDataSet.bkmeans)
	{
           szAssignInfo = nf.format(numassigned);
	   szSecondLine =nf.format(numassigned) +" Genes Assigned";
	}
	else
	{
           szAssignInfo = nf.format(numassigned)+"\t"+nf.format(numexpected)+"\t"+szexp+"\t"+szsig;
           szSecondLine = nf.format(numassigned) +" Genes Assigned; "+nf.format(numexpected)
                       +" Genes Expected; p-value = "+szexp+szsig;
	}

        bounds = fm.getStringBounds(szSecondLine,g2);
        int nwid2 = (int) Math.ceil(bounds.getMaxX());
        g2.drawString(szSecondLine,SCREENWIDTH/2-nwid2/2,70);
	
        NumberFormat nf1 =  NumberFormat.getInstance(Locale.ENGLISH);
        nf1.setMinimumFractionDigits(1);
        nf1.setMaximumFractionDigits(1);
        nf1.setGroupingUsed(false);

	bcheckonly = false;
        if (szSpot != null)
	{
	   bcheckonly = true;
	}

        if (inames != null)
	{
	    bcheckonly = true;
           g2.drawString(szIntersect,SCREENWIDTH/9,90);
	}
        else if ((pgrr != null)&&((theDataSet.tga.szsortcommand.equals("define"))))
	{
	   String szpvalquery = Util.doubleToSz(pgrr.dpval);
           double denriched = pgrr.dgenes - pgrr.dmaxselect*pgrr.dgenestotal/pgrr.dmaxselecttotal;
           String szenriched = enrichedToSz(denriched);          
           String szquery = "Query set "+szprofilecluster+" enrichment uncorrected p-value = "+szpvalquery
	       +" ("+nf1.format(pgrr.dgenes)+"/"+nf1.format(pgrr.dmaxselect)+" vs. "+
	       ((int) pgrr.dgenestotal)+"/"
	       + ((int) pgrr.dmaxselecttotal)+"; "+szenriched+" genes)";

           g2.drawString(szquery,SCREENWIDTH/7,90);
	   bcheckonly = true;
           if ((theDataSet.tga.bcluster)&&(pgrr.dclusterpval < 1.5))
	   {
  	      szpvalquery = Util.doubleToSz(pgrr.dclusterpval);
              denriched = pgrr.dgenescluster - 
                       pgrr.dmaxselectcluster*pgrr.dgenestotal/pgrr.dmaxselecttotal;
              szenriched = enrichedToSz(denriched);
              szquery = "Query set cluster enrichment uncorrected p-value = "+szpvalquery+" ("+
		  nf1.format(pgrr.dgenescluster)+"/"+nf1.format(pgrr.dmaxselectcluster)+" vs. "
		  +((int) pgrr.dgenestotal)+"/"+ ((int) pgrr.dmaxselecttotal)+"; "+szenriched+" genes)";                                  
              g2.drawString(szquery,SCREENWIDTH/7,110);
	   }
	}
	else if ((pgrr != null)&&((theDataSet.tga.szsortcommand.equals("expgo"))||
                                  (theDataSet.tga.szsortcommand.equals("expdefine"))))
	{
	   String szpvalgo = Util.doubleToSz(pgrr.dpval);
           double dprofileexp =theDataSet.expectedassignments[pgrr.nprofile];
           double denriched = pgrr.dgenes - 
                              pgrr.dgenestotal*dprofileexp/pgrr.dmaxselecttotal;
           String szenriched = enrichedToSz(denriched);
	   GoAnnotations.Rec gar; 
           int nyloc;
	   bcheckonly = true;
           if (theDataSet.tga.szsortcommand.equals("expgo"))
	   {
	      gar = (GoAnnotations.Rec) 
                                  theDataSet.tga.htGO.get(theDataSet.tga.szSelectedGO);
              g2.drawString("      "+theDataSet.tga.szSelectedGO+" ("+gar.sztermName+"):",SCREENWIDTH/10,90);
              nyloc =110;
	   }
	   else
	   {
	      nyloc = 90;
	   }

           String szgo =  "        "+szprofileclusterCAP
               +" expected size based enrichment uncorrected p-value = "+szpvalgo
	       +" ("+nf1.format(pgrr.dgenes)+"/"+ nf1.format(dprofileexp)+" vs. "
               +(int)pgrr.dgenestotal+"/"
	       + ((int) pgrr.dmaxselecttotal)+"; "+szenriched+" genes)";
                                   
           g2.drawString(szgo,SCREENWIDTH/10,nyloc);
	}
        else if ((pgrr!=null)&&(theDataSet.tga.szsortcommand.equals("go")))
	{
	   String szpvalgo = Util.doubleToSz(pgrr.dpval);
           double denriched = pgrr.dgenes - pgrr.dmaxselect*pgrr.dgenestotal/pgrr.dmaxselecttotal;
           String szenriched = enrichedToSz(denriched);
	   bcheckonly = true;
	   GoAnnotations.Rec gar = (GoAnnotations.Rec) 
                                  theDataSet.tga.htGO.get(theDataSet.tga.szSelectedGO);
           g2.drawString("       "+theDataSet.tga.szSelectedGO+" ("+gar.sztermName+"):",SCREENWIDTH/10,90);
           String szgo =  "          "+szprofileclusterCAP+
                          " actual size based enrichment uncorrected p-value = "+szpvalgo
	       +" ("+nf1.format(pgrr.dgenes)+"/"+nf1.format(pgrr.dmaxselect)+" vs. "
               + ((int) pgrr.dgenestotal)+"/"
	       + ((int) pgrr.dmaxselecttotal)+"; "+szenriched+" genes)";
                                   
           g2.drawString(szgo,SCREENWIDTH/10,110);

           if ((theDataSet.tga.bcluster)&&(pgrr.dclusterpval < 1.5))
	   {
  	      szpvalgo = Util.doubleToSz(pgrr.dclusterpval);
              denriched = pgrr.dgenescluster - pgrr.dmaxselectcluster*pgrr.dgenestotal/pgrr.dmaxselecttotal;
              szenriched = enrichedToSz(denriched);
              szgo = "          Cluster enrichment uncorrected p-value = "+szpvalgo
		  +" ("+nf1.format(pgrr.dgenescluster)+"/"+nf1.format(pgrr.dmaxselectcluster)+" vs. "
		  +((int) pgrr.dgenestotal)+"/"
		  + ((int) pgrr.dmaxselecttotal)+"; "+szenriched+" genes)";
                                   
              g2.drawString(szgo,SCREENWIDTH/10,130);
	   }
	}

        g2.draw(new Line2D.Double(SPACELEFT,nzero,(numpoints-1)*DELTAX+SPACELEFT,nzero));
        g2.draw(new Line2D.Double(SPACELEFT,nzero-MAXFOLD*DELTAY, SPACELEFT, 
                                     nzero+MINFOLD*DELTAY));     
	int ntickloc=0;
	double dtickloc = nzero;
	double dtickval = 0;

	while (dtickval <= dmaxval+.0000001)
	{
	   g2.draw(new Line2D.Double(SPACELEFT-22,dtickloc,SPACELEFT,(int) dtickloc));
           if (dtickinterval-(int) dtickinterval < .00000001)
	   {
	      g2.drawString(""+nftick.format(dtickval), SPACELEFT-15-3*(npower),(int) dtickloc-2);
	   }
	   else
	   {
	      g2.drawString(""+nftick.format(dtickval), SPACELEFT-18+3*(ndecimalpower-Math.abs(npower)),(int) dtickloc-2);
	   }
	     
	   dtickloc -= dtickinterval*DELTAY/dpowscale;
           dtickval +=dtickinterval;
	}
       
        g2.drawString("Expression Change", SPACELEFT+3,64);

	if (theDataSet.badd0)
	{
           g2.drawString("v(i)",SPACELEFT+3,76);
	}
        else if (theDataSet.btakelog)
	{
           g2.drawString("(log\u2082(v(i)/v(0)))",
                                  SPACELEFT+3,76);
	}
	else
	{
           g2.drawString("(v(i)\u2212v(0))",
                                  SPACELEFT+3,76);
	}

	dtickval = -dtickinterval;
	dtickloc = nzero + dtickinterval*DELTAY/dpowscale;
	while (dtickval >= dminval-.0000001)
	{
	   g2.draw(new Line2D.Double(SPACELEFT-22,dtickloc,SPACELEFT,dtickloc));
           if (dtickinterval-(int) dtickinterval < .00000001)
	   {
	      g2.drawString(""+((int)dtickval), SPACELEFT-19-3*(npower),(int)dtickloc-2);
	   }
	   else
	   {
	      g2.drawString(""+nftick.format(dtickval), SPACELEFT-22+3*(ndecimalpower-Math.abs(npower)),(int) dtickloc-2);
	   }

	   dtickloc += dtickinterval*DELTAY/dpowscale;
	   dtickval -=dtickinterval;
	}
	
        g2.drawString(""+theDataSet.dsamplemins[0], SPACELEFT+5,nzero +20);
        for (int ntick = 1; ntick < numpoints-1; ntick++)
	{
	    g2.draw(new Line2D.Double(SPACELEFT+theDataSet.dwidthunitsCum[ntick]*(numpoints-1)*DELTAX,
                                      nzero-10,SPACELEFT+theDataSet.dwidthunitsCum[ntick]*(numpoints-1)*DELTAX,nzero+10));
            g2.drawString("" +theDataSet.dsamplemins[ntick],(int)(SPACELEFT+theDataSet.dwidthunitsCum[ntick]
								  *(numpoints-1)*DELTAX),nzero+20);
        }
        int ntick = numpoints - 1;
	g2.draw(new Line2D.Double(SPACELEFT+ntick*DELTAX,nzero-10,SPACELEFT+ntick*DELTAX,nzero+10));
        g2.drawString("" +theDataSet.dsamplemins[ntick],SPACELEFT+ntick*DELTAX,nzero+20);
        g2.drawString("TIME",SPACELEFT+ntick*DELTAX+2,nzero-5);

	noffset =60;
        tablewid = 157;
        nmidpoint = (SCREENWIDTH + SPACELEFT -SPACERIGHT)/2;
        int nbstart= (int) (SCREENWIDTH/2-2*tablewid-1.5*BSPACING);
        if (bshowcluster)
        {
	  profiletablex = nbstart;
	  profilegox = profiletablex+tablewid+BSPACING;
	  clustertablex =profilegox+tablewid+BSPACING;
	  clustergox = clustertablex+tablewid+BSPACING;
	}
        else
	{
	    profiletablex = nbstart+tablewid+BSPACING;
	    profilegox = profiletablex+tablewid+BSPACING;
	}

        tableheight = 20;
        if ((szSpot!=null)||((inames == null)&&
           (!theDataSet.tga.szsortcommand.equals("define"))&&
	   (!theDataSet.tga.szsortcommand.equals("expdefine"))))
        {
           tabley = SCREENHEIGHT-SPACEBOT-3+tableheight+4;
	   nonlyy = tabley - tableheight-5;
	}
	else
        {
	    querytabley = SCREENHEIGHT-SPACEBOT-3;
     	    nonlyy = querytabley - tableheight-5;
	    tabley = querytabley +tableheight +4;
	}
 
   
        if ((inames == null)&&(pgrr!=null)&&  (pgrr.dgenes>0)&&
	    ((theDataSet.tga.szsortcommand.equals("go"))||
	     (theDataSet.tga.szsortcommand.equals("expgo"))))
	{        
           ngosorttabley =  nonlyy - tableheight-5; 
           bgosorttable = true;
	   drawgosorttable();      
	}
	else
	{
	    bgosorttable = false;
	}

        drawprofiletable(ST.buttonColor);
        drawprofilego(ST.buttonColor);

        if (inames != null)
        {
           drawintersecttable(ST.buttonColor);
           drawintersectgo(ST.buttonColor);
        }            
        else if ((pgrr != null)&&((theDataSet.tga.szsortcommand.equals("define"))||
				  (theDataSet.tga.szsortcommand.equals("expdefine"))))
        {
           drawprofilequerytable(ST.buttonColor);
           drawprofilequerygo(ST.buttonColor);
        }

	if (bcheckonly)
	{
	   drawonlyorder();
	}

        if (bshowcluster)
        {
            drawclustertable(ST.buttonColor);
	    drawclustergo(ST.buttonColor);
            if ((pgrr != null)&&((theDataSet.tga.szsortcommand.equals("define"))||
				 (theDataSet.tga.szsortcommand.equals("expdefine"))))
	    {
              drawclusterquerytable(ST.buttonColor);
              drawclusterquerygo(ST.buttonColor);
	    }
	}


        int ncolor = 0;
	Random theRandom = new Random();
	for (int nindex =0; nindex < numuniquegenes; nindex++)
	{
            int nrow =  ((Integer) currProfileAssignments.get(nindex)).intValue();
 
  	    if (ncolor >= colorlist.size())
	    {
		colorlist.add(new Color(theRandom.nextInt(226),theRandom.nextInt(226),theRandom.nextInt(226)));
	    }

            g2.setColor((Color) colorlist.get(ncolor));
            ncolor++;
	    if ((!bonlyorder)|| ((szSpot !=null)&&(theDataSet.probenames[nrow].equals(szSpot)))||
		((inames!=null)&&(inames.contains(theDataSet.genenames[nrow])))||
		((szSpot==null)&&(inames==null)&&
                 (theDataSet.tga.isOrder(theDataSet.genenames[nrow]))))
	    {
              
	       for (int ncol = 0; ncol < numpoints-1; ncol++)
	       {
                  if ((missing[nrow][ncol]>0)&&(missing[nrow][ncol+1]>0))
	          {
		      g2.draw(new Line2D.Double(
						SPACELEFT+theDataSet.dwidthunitsCum[ncol]*DELTAX*(numpoints-1),
                                                nzero-DELTAY*genevalues[nrow][ncol]/dpowscale,
                                                SPACELEFT+theDataSet.dwidthunitsCum[ncol+1]*DELTAX*(numpoints-1),
                                                nzero-DELTAY*genevalues[nrow][ncol+1]/dpowscale));
	          }
	       }
	   }
	}
		
    }

    /**
     * Renders the interface by getting the interface option settings and then calling plotgenevalues
     */
    public void paint(Graphics g) 
    {
       if (thegeneplotpanel == null)
       {
          bautomatic =false;
       }
       else
       {
          bautomatic = thegeneplotpanel.automaticButton.isSelected();
          try
          {
             thegeneplotpanel.thespinnerminy.commitEdit();
	     thegeneplotpanel.thespinnermaxy.commitEdit();
	     thegeneplotpanel.thespinnertick.commitEdit();
	  }
	  catch(ParseException pex)
	  {
	     System.out.println("Warning could not correctly parse y-axis parameters");
	  }
	  dminfixed = ((Double) thegeneplotpanel.thespinnerminy.getValue()).doubleValue();
          dmaxfixed = ((Double) thegeneplotpanel.thespinnermaxy.getValue()).doubleValue();
          dtickinterval = ((Double) thegeneplotpanel.thespinnertick.getValue()).doubleValue();
       }

       if (((theDataSet.tga.szsortcommand.equals("id"))||
            (theDataSet.tga.szsortcommand.equals("sig"))||
            (theDataSet.tga.szsortcommand.equals("num"))||
            (theDataSet.tga.szsortcommand.equals("default"))||
	    (theDataSet.tga.szsortcommand.equals("exp")))&&((szSpot==null)&&(inames==null)))
	{
           bonlyorder = false;
	}

        g2 = (Graphics2D) getGraphics();
        if ((theDataSet.tga.tpgr != null)&&(ranklookup!=null))
	{
           pgrr = theDataSet.tga.tpgr[ranklookup[nprofile]];   
	} 
	else
	{
           pgrr =null;
        }

        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

       
        double[][] genevalues = theDataSet.data; 
        plotgenevalues(g2,theDataSet.countassignments[nprofile],theDataSet.expectedassignments[nprofile],
                        theDataSet.pvaluesassignments[nprofile],genevalues,theDataSet.pmavalues,theDataSet.bestassignments,
                         theDataSet.modelprofiles[nprofile],theDataSet.significantnum,theDataSet.profilesAssigned[nprofile]);        

    }
}

