package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.chromviewer.*;
import edu.cmu.cs.sb.core.*;
import java.util.*;
import javax.swing.*;
import java.awt.event.*;

/**
 * Class encapsulates information on the comparison of two STEM data sets
 * focusing on identifying pairs of profiles with significant intersections 
 * between them.
 */
public class CompareInfo 
{
    /**
     * The main reference data set
     */
    STEM_DataSet origset;

    /**
     * The comparison data set
     */
    STEM_DataSet comparesetfmnel;

    /**
     * The p-value threshold for a significant interestection
     */
    double dmaxpval;

    /**
     * The mininum number of genes in a significant intersection
     */
    int nminnumgenes;

    /**
     * Arrays of CompareInfoRow about significant intersections for one priofile
     */
    ArrayList sigrows;

    /**
     * The version of sigrows transposed when swapping which data set is positioned
     * as the reference and which is the comparison
     */
    ArrayList sigrowsswap;

    /**
     * The maingui window for the comparison data
     */
    MAINGUI2 compareframe;


    /**
     * Launches a maingui for the comparison data and finds the significant pairs of intersections
     */
    public CompareInfo(STEM_DataSet origset, String szcompare, 
		       String szmaxpval, String szminnumgenes,
                       Vector repeatnames, boolean balltime, ChromFrame cf) throws Exception
    {

       this.origset = origset;

       dmaxpval = Double.parseDouble(szmaxpval);
       nminnumgenes = Integer.parseInt(szminnumgenes);
       sigrows = new ArrayList();
       //signames = new ArrayList();

       //builds a STEM comparison data set for this new data set, using the same parameters as the original set
       comparesetfmnel = ST.buildset(
		     cf.genomeParser.szchromval,
		    origset.tga.szxrefval,
                    szcompare, origset.tga.szGoFile, 
                    origset.tga.szGoCategoryFile, origset.nmaxmissing,
                    origset.dthresholdvalue, origset.dmincorrelation,
                    origset.dminclustdist, origset.alpha,
                    origset.dpercentileclust, origset.nmaxchange,
                    origset.nmaxprofiles, origset.dmaxcorrmodel,
                    origset.nsamplesgene, origset.tga.nsamplespval, 
                    origset.nsamplesmodel, origset.tga.nmingo, 
                    origset.nfdr, origset.tga.nmingolevel,
                    origset.tga.szextraval, balltime, repeatnames,
		    origset.btakelog,origset.bspotincluded,
                    origset.badd0,origset.tga.szcategoryIDval,
                    origset.tga.szevidenceval,origset.tga.sztaxonval,
                    origset.tga.bpontoval,origset.tga.bcontoval,origset.tga.bfontoval,
                    origset.tga.brandomgoval,origset.bkmeans,
                    origset.bmaxminval,origset.ballpermuteval,
		    origset.tga.szorganismsourceval, origset.tga.szxrefsourceval,
		    cf.genomeParser.szchromsourceval);

       if (comparesetfmnel.numcols != origset.numcols)
       {
          throw new IllegalArgumentException("Compare data set must have same number of columns as original, "+
                                  "expecting "
                                   +origset.numcols+" found "+comparesetfmnel.numcols+" in the comparison set");
       }
 
       //launches an interface for this comparison dataset
       compareframe = new MAINGUI2(comparesetfmnel);
       compareframe.cf = cf;
       edu.umd.cs.piccolo.PCanvas.CURRENT_ZCANVAS = null;
 
       compareframe.setLocation(25,60);
       compareframe.setDefaultCloseOperation(JFrame.DISPOSE_ON_CLOSE);
       compareframe.addWindowListener(new WindowAdapter() {
          public void windowClosing(WindowEvent we) {
		    compareframe.closeSortWindows();
		}
                 public void windowClosed(WindowEvent we) {
                     Thread t = new Thread (new Runnable() { 
                                              public void run() { 
		                                 System.gc();          
                                              } 
                                          } 
                                    ); 
                     t.start(); 
		 }
	      });
	 
        compareframe.setVisible(true);
        findsigcompare();

    }  

    /**
     * Returns the number of common elements in hm1 and hm2
     */
    private int countintersect(HashMap hm1, HashMap hm2)
    {
	Iterator em1 = hm1.keySet().iterator();
        int ncount = hm2.size();
        while (em1.hasNext())
	{
	    String sz1 = (String) em1.next();
            if (hm2.get(sz1) == null)
	    {
		ncount++;
	    }
	}
        return ncount;
    }

    /**
     * Finds significant pairs of profiles from the two data sets interms of an intersection
     * exceeding nminnumgenes and p-value less than dmaxpval
     */
    public void findsigcompare()
    {

        int nsize1, nsize2;
        ArrayList[] sigprofilesswapArray = new ArrayList[comparesetfmnel.modelprofiles.length];
        ArrayList[] sigprofilesswapnames = new ArrayList[comparesetfmnel.modelprofiles.length];
        double[] mincorrswap = new double[comparesetfmnel.modelprofiles.length];
        double[] minpvalswap = new double[comparesetfmnel.modelprofiles.length];

        for (int nindex = 0; nindex < sigprofilesswapArray.length; nindex++)
        {
	   sigprofilesswapArray[nindex] = new ArrayList();
	   sigprofilesswapnames[nindex] = new ArrayList();
           mincorrswap[nindex] = 1;
           minpvalswap[nindex] = 1;
	}

        int numintersect = countintersect(origset.tga.htGeneNames, comparesetfmnel.tga.htGeneNames);
	System.out.println("Number of genes in the union of the original and comparison set is "+numintersect);

        for (int nprofileindex = 0; nprofileindex < origset.modelprofiles.length; nprofileindex++)
	{
            HashSet htprofilegenes = new HashSet();
            nsize1 =origset.profilesAssigned[nprofileindex].size();
            for (int ngeneindex = 0; ngeneindex < nsize1; ngeneindex++)
	    {
                int nrealgeneindex = 
                          ((Integer) origset.profilesAssigned[nprofileindex].get(ngeneindex)).intValue();
                htprofilegenes.add(origset.genenames[nrealgeneindex]);
	    }
               
            ArrayList sigprofiles = new ArrayList();
            ArrayList sigprofilesnames = new ArrayList();   
            double dminpval = 1;
            double dmincorr = 1;
            for (int nprofileindex2 = 0; nprofileindex2 < comparesetfmnel.modelprofiles.length; nprofileindex2++)
	    {
               HashSet intersectnames = new HashSet();  
               double dmatch = 0;
               nsize2 = comparesetfmnel.profilesAssigned[nprofileindex2].size();

	       for (int ngeneindex = 0; ngeneindex < nsize2; ngeneindex++)
	       {
                  int nrealgeneindex = 
                        ((Integer) comparesetfmnel.profilesAssigned[nprofileindex2].get(ngeneindex)).intValue();
                  String szgene = comparesetfmnel.genenames[nrealgeneindex];
                  if (htprofilegenes.contains(szgene))
		  {
                      intersectnames.add(szgene);
                      dmatch +=1.0/comparesetfmnel.bestassignments[nrealgeneindex].size();
                  }
	       }

              double dpval =StatUtil.hypergeometrictail((int) Math.ceil(dmatch-1),nsize1,
			   (int)Math.floor(numintersect-nsize1),nsize2);
               if ((dmatch >=nminnumgenes)&&(dpval <= dmaxpval))
	       {
		  if (dpval < dminpval)
		  {
		     dminpval = dpval;
		  }

                  if (dpval < minpvalswap[nprofileindex2])
		  {
		      minpvalswap[nprofileindex2] = dpval;
		  }
                 
                  double dcorrval = Util.correlation(origset.modelprofiles[nprofileindex],
						      comparesetfmnel.modelprofiles[nprofileindex2]);
                  if (dcorrval < dmincorr)
		  {
                      dmincorr = dcorrval; 
		  }

                  if (dcorrval < mincorrswap[nprofileindex2])
		  {
		      mincorrswap[nprofileindex2] = dcorrval;
		  }

                  sigprofiles.add(new CompareInfoRec(dmatch, dpval,dcorrval, nprofileindex2,intersectnames));
                  sigprofilesswapArray[nprofileindex2].add(new CompareInfoRec(dmatch,dpval,dcorrval,
									  nprofileindex,intersectnames));
	       }
	    }

            if (sigprofiles.size()>=1)
	    {
                sigrows.add(new CompareInfoRow(nprofileindex, dmincorr, dminpval,sigprofiles));
	    }
	}

        sigrowsswap = new ArrayList();
        for (int nprofileindex2 = 0; nprofileindex2 < sigprofilesswapArray.length; nprofileindex2++)
	{
            if  (sigprofilesswapArray[nprofileindex2].size() >= 1)
	    {
		sigrowsswap.add(new CompareInfoRow(nprofileindex2,mincorrswap[nprofileindex2],
			      minpvalswap[nprofileindex2],sigprofilesswapArray[nprofileindex2]));
	    }
	}
    }


    /**
     * Information on an intersection match: the number in the match, the p-value,
     * the correlation of the profiles, the profile ID, and the genes in the intersection.
     */
    static class CompareInfoRec
    {
	double dmatch;
        double dpval;
        double dcorrval;
        int nprofile;
        HashSet inames;

        CompareInfoRec(double dmatch, double dpval, double dcorrval, int nprofile, HashSet inames)
	{
	    this.dmatch = dmatch;
            this.dpval  = dpval;
            this.dcorrval = dcorrval;
            this.nprofile = nprofile;
            this.inames  = inames;
	}
    }

    /**
     *For a row of intersections: the profile ID from one data set, the list of significant intersections in
     * signprofiles, and the minimum correlation and p-value with any of these significant intersections
     */
   static class CompareInfoRow
   {
       int nprofile;
       double dmincorr;
       double dminpval;
       ArrayList sigprofiles;

       CompareInfoRow(int nprofile,double dmincorr, double dminpval, ArrayList sigprofiles)
       {
	   this.dmincorr = dmincorr;
	   this.dminpval = dminpval;
           this.nprofile = nprofile;
           this.sigprofiles = sigprofiles;
       }
   }

}
