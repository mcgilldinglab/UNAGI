package edu.cmu.cs.sb.stem;

import edu.cmu.cs.sb.core.*;
import java.io.*;
import java.util.*;
import java.text.*;

/**
 * Class implementing clustering methods implemented in STEM
 */
public class STEM_DataSet extends DataSetCore
{

    public static double DSAME = 1.2;
    public static double FLOATERROR = 0.0000001;
    public static int ALLPERMSTHRESH = 9;

    STEM_GoAnnotations tga;

    boolean bkmeans = false;

    double[] countassignments, expectedassignments,pvaluesassignments;

    ArrayList[] profilesAssigned;
    double dmaxminclustdist;

    int nsamplesgene = -1;
    long nsamplesmodel;
    STEM_DataSet otherset = null;
    boolean bothersetorigset = true;

    ArrayList[] bestassignments;
    double[][] modelprofiles;

    int nfdr;  //0 none, 1 fdr, 2 bonferonni
    boolean[] significantnum;


    double[] dwidthunitsCum;
    double alpha;

    boolean ballpermuteval; //this is whether to hold out 0 or not (ballperm
    int nmaxprofiles;

    ArrayList clustersofprofilesnum = new ArrayList(); 
    double dminclustdist=0.7;

    double dpercentileclust =0;
    double dmaxcorrmodel = DSAME;
    int nmaxchange = 1;

    /**
     *The first index is the profile, the second index is the time point.
     *profileavg[i][j] is the average value of all genes assigned to 
     *the profile i at the (j+1)^th time point.
     *The first time point which is always 0 is not included.
     */
    public double[][] profileavg; 

    /**
     *The first index is the profile, the second index is the time point.
     *profileavg[i][j] is the standard deviation of the value of all genes assigned to 
     *the profile i at the (j+1)^th time point.
     *The first time point which is always 0 is not included.
     */
    public double[][] profilestd;

    /**
     *The first index is the profile, the second index is the time point.
     *profileavg[i][j] is the square of the value of all genes assigned to 
     *the profile i at the (j+1)^th time.
     *The first time point which is always 0 is not included.
     */
    public double[][] profilesumsq;

    /**
     * Constructor that simply copies all fields in theDataSetCore and theSTEM_DataSet
     */
    public STEM_DataSet(DataSetCore theDataSetCore,STEM_DataSet theSTEM_DataSet)
    {
	super(theDataSetCore);
	this.tga = theSTEM_DataSet.tga;
	this.bkmeans = theSTEM_DataSet.bkmeans;
	this.countassignments =theSTEM_DataSet.countassignments;
	this.expectedassignments = theSTEM_DataSet.expectedassignments;
	this.pvaluesassignments = theSTEM_DataSet.pvaluesassignments;
	this.profilesAssigned = theSTEM_DataSet.profilesAssigned;
	this.dmaxminclustdist = theSTEM_DataSet.dmaxminclustdist;
	this.nsamplesgene = theSTEM_DataSet.nsamplesgene;
	this.nsamplesmodel = theSTEM_DataSet.nsamplesmodel;
	this.otherset = theSTEM_DataSet.otherset;
	this.bothersetorigset = theSTEM_DataSet.bothersetorigset;
	this.bestassignments = theSTEM_DataSet.bestassignments;
	this.modelprofiles = theSTEM_DataSet.modelprofiles;
	this.profileavg = theSTEM_DataSet.profileavg;
	this.profilestd = theSTEM_DataSet.profilestd;
	this.profilesumsq = theSTEM_DataSet.profilesumsq;
	this.alpha = theSTEM_DataSet.alpha;
	this.ballpermuteval = theSTEM_DataSet.ballpermuteval;
	this.nmaxprofiles = theSTEM_DataSet.nmaxprofiles;
	this.clustersofprofilesnum = theSTEM_DataSet.clustersofprofilesnum;
	this.dminclustdist = theSTEM_DataSet.dminclustdist;
	this.nfdr = theSTEM_DataSet.nfdr;
	this.significantnum = theSTEM_DataSet.significantnum;
	this.dpercentileclust = theSTEM_DataSet.dpercentileclust;
	this.dmaxcorrmodel =  theSTEM_DataSet.dmaxcorrmodel;
	this.nmaxchange = theSTEM_DataSet.nmaxchange;
	this.dwidthunitsCum =  theSTEM_DataSet.dwidthunitsCum;
    }
    //////////////////////////////////////////////////////////////////////////////////

    /**
     * Constructor for k-means clustering
     */
    public STEM_DataSet(String szInputFile,int nmaxmissing, 
                   double dthresholdvalue, double dmincorrelation, 
                   int nmaxprofiles, int nmaxchange, boolean btakelog,
                   boolean bspotincluded,boolean brepeatset,boolean badd0,
                   boolean bmaxminval,boolean bfullrepeat)
                               throws IOException, FileNotFoundException, IllegalArgumentException
    {
	//k-means version
	this.szInputFile = szInputFile;
	this.bfullrepeat = bfullrepeat;
	bkmeans = true;
	this.nmaxmissing = nmaxmissing;
	this.dthresholdvalue = dthresholdvalue;
	this.dmincorrelation = dmincorrelation;
	this.nmaxprofiles = nmaxprofiles;
	this.nmaxchange = nmaxchange;
	this.badd0 = badd0;
	this.btakelog = btakelog;
	this.bspotincluded = bspotincluded;
	this.bmaxminval = bmaxminval;

	dataSetReader(szInputFile, nmaxmissing,
                    dthresholdvalue, dmincorrelation,
		    btakelog, bspotincluded, brepeatset, badd0);
    }

    /**
     * Constructor for a STEM method clustering
     */
    public STEM_DataSet(String szInputFile, int nmaxmissing,
                   double dminclustdist, double dthresholdvalue, 
                   double dmincorrelation, double dalphaval,
                   double dpercentileclust,int nmaxchange,int nmaxprofiles, double dmaxcorrmodel, 
                   int nsamplesgene, long nsamplesmodel,double[][] modelprofiles, int nfdr, 
                   boolean btakelog,boolean bspotincluded, boolean brepeatset,boolean badd0,
                   boolean bmaxminval, boolean ballpermuteval, boolean bfullrepeat)
   	                 throws IOException, FileNotFoundException, IllegalArgumentException
    {
	this.bfullrepeat = bfullrepeat;
	this.szInputFile = szInputFile;
	this.nmaxmissing = nmaxmissing;
        this.nsamplesgene  = nsamplesgene;
        this.nsamplesmodel = nsamplesmodel;
        this.dthresholdvalue = dthresholdvalue;
        this.dmincorrelation = dmincorrelation;
        this.dminclustdist = dminclustdist;
        this.alpha = dalphaval;
        this.dpercentileclust = dpercentileclust;
        this.nmaxchange = nmaxchange;	
        this.nmaxprofiles = nmaxprofiles;		
        this.dmaxcorrmodel = dmaxcorrmodel;
	this.nfdr = nfdr;     
	this.btakelog = btakelog;
	this.bspotincluded = bspotincluded;
	this.badd0 = badd0;
	this.bmaxminval = bmaxminval;
        this.ballpermuteval = ballpermuteval;

	dataSetReader(szInputFile, nmaxmissing,
                    dthresholdvalue, dmincorrelation, 
		    btakelog, bspotincluded, brepeatset, badd0);

        if (modelprofiles == null)
	{
           generatemodelprofiles(numcols,nmaxchange);
	}
        else
	{
	    this.modelprofiles = modelprofiles;
	}
    }

    //////////////////////////////////////////////////////////////////////////////////
    /**
     * Counts the number of genes assigned to each profile
     */
    public void tallyassignments()
    {
        countassignments = new double[modelprofiles.length];
	for (int nindex = 0; nindex < modelprofiles.length; nindex++)
	{
            countassignments[nindex] = 0;
	}

        for (int nrow = 0; nrow < numrows; nrow++)
	{
            ArrayList assignments = bestassignments[nrow];
            int numassigned = assignments.size();
            double dweight = 1/(double) numassigned;
            for (int nindex = 0; nindex < numassigned; nindex++)
	    {
		countassignments[((Integer) assignments.get(nindex)).intValue()] += dweight;
	    }
	}
    }


    /////////////////////////////////////////////////////////////
    /**
     * Computes the average and standard deviation of the expression values of genes
     * assigned to the same profile
     */
    public void computeprofilestats()
    {
       int nnonzero = modelprofiles[0].length - 1;
       profileavg = new double[modelprofiles.length][nnonzero];
       profilestd = new double[modelprofiles.length][nnonzero];
       profilesumsq = new double[modelprofiles.length][nnonzero];

       for (int nprofile = 0; nprofile < modelprofiles.length; nprofile++)
       {
          ArrayList geneList = profilesAssigned[nprofile];
     
          double[] dsumvals = new double[nnonzero];
          double[] dsumvalssq = new double[nnonzero];
          double dweightsum = 0;
          int numvals = 0;
               
          for (int nindex = 0; nindex < nnonzero; nindex++)
          {
	     dsumvals[nindex] = 0;
             dsumvalssq[nindex] = 0;
	  }

          int nlistsize = geneList.size();

          for (int nindex = 0; nindex < nlistsize; nindex++)
          {
	     int ngene = ((Integer) geneList.get(nindex)).intValue();
             double dweight = 1.0/(double) bestassignments[ngene].size();
             for (int ntime = 0; ntime < nnonzero; ntime++)
             {
	        dsumvals[ntime] += dweight*data[ngene][ntime+1];
                dsumvalssq[ntime] += dweight*Math.pow(data[ngene][ntime+1],2); 
	     }
             numvals++;
             dweightsum += dweight;
          }

          for (int ntime = 0; ntime < nnonzero; ntime++)
          {
             if (dweightsum > 0)
	     {
	        profileavg[nprofile][ntime] = dsumvals[ntime]/dweightsum;
             }
             else
       	     {
	        profileavg[nprofile][ntime] = 0;
	     }
             
             profilesumsq[nprofile][ntime] = dsumvalssq[ntime];
                
             if (numvals > 1)
      	     {
                double dval = (double) numvals / (double) (numvals-1)* 
                                   (dsumvalssq[ntime]/dweightsum - 
		                      Math.pow(dsumvals[ntime]/dweightsum,2));
                profilestd[nprofile][ntime] = Math.sqrt(dval); 
	     }
             else
	     {
	        profilestd[nprofile][ntime] = 0;
	     }
	  }  
       }  
    }

    ////////////////////////////////////////////////////////////////////

    /**
     * Computes the p-value for genes assigned to the same profile
     */
    public void computePvaluesAssignments()
    {
       significantnum = new boolean[expectedassignments.length];
       pvaluesassignments = new double[expectedassignments.length];
       for (int nindex = 0; nindex < expectedassignments.length; nindex++)
       {
          double dprob = expectedassignments[nindex]/numrows;
          pvaluesassignments[nindex] = 
	      StatUtil.binomialtail((int) Math.ceil(countassignments[nindex]-1), numrows, dprob);
       }

       if (nfdr==1)
       {
          double[] pvalcopy = new double[modelprofiles.length];
          System.arraycopy(pvaluesassignments,0,pvalcopy,0,modelprofiles.length);

          Arrays.sort(pvalcopy);
	  int npvalindex = 0;
          double dthresh=0;
          while (pvalcopy[npvalindex] < (npvalindex+1)*alpha / (modelprofiles.length))
	  {
	     dthresh = pvalcopy[npvalindex];
             npvalindex++;
	  }

          for (int nsigindex = 0; nsigindex < modelprofiles.length; nsigindex++)
          {
	     significantnum[nsigindex] = pvaluesassignments[nsigindex] <= dthresh;
	  }
       }
       else
       {
          double dcorrectedalpha;
	  if (nfdr == 0)
          {
	     dcorrectedalpha = alpha;
	  }
          else
	  {
	     dcorrectedalpha = alpha / modelprofiles.length;		
          }
	 
          for (int nsigindex = 0; nsigindex < expectedassignments.length; nsigindex++)
	  {
             significantnum[nsigindex] = (pvaluesassignments[nsigindex] < dcorrectedalpha)
                                         &&(countassignments[nsigindex]>1);
	  }
       }
    }

    ///////////////////////////////////////////////////////////////
    /**
     * Compares two profiles records based on their dprofiledist variable
     */
    public static class Profilerecdistcomparator implements Comparator
    {
        public int compare(Object o1, Object o2)
	{
            ProfileRecDist rec1 = (ProfileRecDist) o1;
            ProfileRecDist rec2 = (ProfileRecDist) o2;

            if (rec1.dprofiledist < rec2.dprofiledist)
	    {
		return 1;
	    }
            else if (rec1.dprofiledist > rec2.dprofiledist)
	    {
                return -1;
	    }  
            else
	    {
		return 0;
	    }
	}
    }

    /////////////////////////////////////////////////////////////////

    /**
     *Class implements a record storing a ProfileRec, theProfileRec
     *and a distance value, dprofiledist 
     */
    static class ProfileRecDist
    {
	double dprofiledist;
        ProfileRec theProfileRec;

        ProfileRecDist(ProfileRec theProfileRec, double dprofiledist)
	{
            this.theProfileRec = theProfileRec;
            this.dprofiledist  = dprofiledist;
	}
    }

    ///////////////////////////////////////////////////////////////

    /**
     *Class implements a record for a profile, storing its ID,
     *the number of genes assigned, the number of genes expected, 
     *and the uncorrected p-value of the number of genes assigned versus expected
     */
    public static final class ProfileRec
    {
        public int nprofileindex;
        public double dnumgenes;
        public double dnumexpected;
        public double dpvalue;

        ProfileRec(int nprofileindex, double dnumgenes, double dnumexpected, double dpvalue)
	{
	    this.nprofileindex = nprofileindex;
            this.dnumgenes = dnumgenes;
            this.dnumexpected = dnumexpected;
            this.dpvalue = dpvalue;
	}
	
        public Object clone()
	{
            return new ProfileRec(nprofileindex, dnumgenes, dnumexpected, dpvalue);
	}
    }

    //////////////////////////////////////////////////////////////////////////////////////
    /**
     * Implements the k-means clustering method
     */
    public void kmeans()
    {
	//NOTE:
	//maxprofiles is nK  reusing variable not used for STEM method
	//maxchange is numreps  reusing variable not used for STEM method

	if (nmaxprofiles > data.length)
	{
	    throw new IllegalArgumentException("More clusters ("+nmaxprofiles+
                                             ") requested than genes passing filter ("+data.length+")");
	}

	double[][] clusterdata;

        clusterdata = new double[data.length][data[0].length];

	for (int ngene = 0; ngene < clusterdata.length; ngene++)
	{
           for (int nindex = 0; nindex < clusterdata[ngene].length; nindex++)
           {
	      clusterdata[ngene][nindex] = data[ngene][nindex];
           }
	}

	int[] nonmissing = new int[pmavalues.length];
	int[] missing = new int[pmavalues.length];

	double[][] currprofiles = new double[nmaxprofiles][data[0].length];
	double[][] bestcurrprofiles=null;

	ArrayList[] currassignmentAL = new ArrayList[data.length];
	ArrayList[] assignmentAL = null;
	double dminscore = 0;

  	Random xrand = new Random(2211);
	for (int nrep = 0; nrep < nmaxchange;nrep++)
	{
	   int nonmissingindex = 0;
	   int nmissingindex = 0;
	   for (int ngene = 0; ngene < nonmissing.length; ngene++)
	   {
	      boolean bnomissing = true;
	      int nindex = 0;
	      while ((bnomissing) && (nindex < pmavalues[ngene].length))
	      {
	         if (pmavalues[ngene][nindex] == 0)
		 {
		    bnomissing = false;
		 }
	         nindex++;
	      }

	      if (bnomissing)
	      {
	         nonmissing[nonmissingindex] = ngene;
		 nonmissingindex++;
	      }
	      else
	      {
	         missing[nmissingindex] = ngene;
	 	 nmissingindex++;
	      }
	   }



	   int nrandom = Math.min(currprofiles.length,nonmissingindex);
           int[] rindex = new int[currprofiles.length];
           for (int nindex = 0; nindex < nrandom; nindex++)
	   {
              rindex[nindex] = nindex;
           }

           //drawing nrandom elements from a set of numtotalgenes elements
           //where each element is equally likely
           for (int nindex = nrandom; nindex < nonmissingindex; nindex++)
	   {
              if (xrand.nextDouble() < ((double) nrandom/(double) (nindex+1)))
              {
	         rindex[(int) Math.floor(nrandom*xrand.nextDouble())] = nindex;
              }
           }

	   if (nonmissingindex < currprofiles.length)
	   {
	       //need to finish random centers with some missing
	      int nrandommissing = currprofiles.length - nonmissingindex;
              for (int nindex = 0; nindex < nrandommissing; nindex++)
 	      {
                 rindex[nindex+nonmissingindex] = nindex;
              }

              //drawing nrandom elements from a set of numtotalgenes elements
              //where each element is equally likely
              for (int nindex = nrandommissing; nindex < nmissingindex; nindex++)
	      {
                 if (xrand.nextDouble() < ((double) nrandommissing/(double) (nindex+1)))
                 {
	            rindex[(int) Math.floor(nrandommissing*xrand.nextDouble())+nonmissingindex] = nindex;
                 }
              }

	      for (int nindex = nonmissingindex; nindex < rindex.length; nindex++)
              {
	         for (int nindex2 = 0; nindex2 <pmavalues[rindex[nindex]].length; nindex2++)
	         { 
		    if (pmavalues[rindex[nindex]][nindex2]== 0)
		    {
		       //randomly sets missing data values to a value between -1 and 1
		       clusterdata[rindex[nindex]][nindex2] = 2*xrand.nextDouble()-1;
		    }
	         }
	      } 
	   }

  	   for (int ncenter = 0; ncenter < nrandom; ncenter++)
	   {
              int nrindex = rindex[ncenter];
	      for (int ncol = 0; ncol < currprofiles[0].length; ncol++)
	      {
	         currprofiles[ncenter][ncol] = clusterdata[nonmissing[nrindex]][ncol];
	      }
	   }


  	   for (int ncenter = nrandom; ncenter < currprofiles.length; ncenter++)
	   {
              int nrindex = rindex[ncenter];
	      for (int ncol = 0; ncol < currprofiles[0].length; ncol++)
	      {
	         currprofiles[ncenter][ncol] = clusterdata[missing[nrindex]][ncol];
	      }
	   }

   	   boolean bchange = true;

	   //int[] assignment = new int[data.length];
	   double[] mindist = new double[data.length];
	   double[][] dsum    = new double[nmaxprofiles][data[0].length];
	   double[][] dcount  = new double[nmaxprofiles][data[0].length];

	   while (bchange)
	   {
	      bchange = false;
	      for (int nindex = 0; nindex < dsum.length; nindex++)
	      {
	         for (int ncol = 0; ncol < dsum[0].length; ncol++)
	         {
	            dsum[nindex][ncol] = 0;
		    dcount[nindex][ncol] = 0;
	         }
	      }

  	      double ddist;
	      for (int nrow = 0; nrow < clusterdata.length; nrow++)
	      {
		 ArrayList nbestdistindexAL = new ArrayList();
		 nbestdistindexAL.add(new Integer(0));
                 mindist[nrow] = Util.distortion(clusterdata[nrow], currprofiles[0],pmavalues[nrow]);

	         for (int ncenterindex = 1; ncenterindex < nmaxprofiles; ncenterindex++)
	         {
		    ddist = Util.distortion(clusterdata[nrow], currprofiles[ncenterindex],pmavalues[nrow]);

		    if (ddist < mindist[nrow])
		    {
		       mindist[nrow] = ddist;
		       nbestdistindexAL = new ArrayList();
		       nbestdistindexAL.add(new Integer(ncenterindex));
		    }
		    else if (ddist == mindist[nrow])
		    {
			nbestdistindexAL.add(new Integer(ncenterindex));
		    }
	         }

                 for (int ncol = 0; ncol < dsum[0].length; ncol++)
	         {
		    if (pmavalues[nrow][ncol]!=0)
		    {
		       int nsize = nbestdistindexAL.size();
		       double dweight = 1.0/ nsize;
		       for (int ni = 0; ni < nsize; ni++)
		       {
			   int ncenter = ((Integer) nbestdistindexAL.get(ni)).intValue();
			   dsum[ncenter][ncol] += dweight*clusterdata[nrow][ncol];
			   dcount[ncenter][ncol]+= dweight;
		       }
		    }
	         }

		 int nel =0;
		 int nsize = nbestdistindexAL.size();
		 boolean bsame =((currassignmentAL[nrow] != null)&&(currassignmentAL[nrow].size() == nsize));
		 while ((bsame)&&(nel < nsize))
		 {
		    if (nbestdistindexAL.get(nel).equals(currassignmentAL[nrow].get(nel)))
		    {
			nel++;
		    }
		    else
		    {
			bsame = false;
		    }
		 }

		 if (!bsame)
		 {
		     currassignmentAL[nrow] = nbestdistindexAL;
		     bchange = true;
		 }	       
	      }

  	      for (int nindex = 0; nindex < dsum.length; nindex++)
	      {
	         for (int ncol = 0; ncol < dsum[0].length; ncol++)
	         {
		    if (dcount[nindex][ncol] > 0)
		    {
		       currprofiles[nindex][ncol] = dsum[nindex][ncol]/dcount[nindex][ncol];
		    }
	         }
	      }
	   }


	   double dtotscore = 0;
	   for (int nrow = 0; nrow < clusterdata.length; nrow++)
	   {
	       int nsize = currassignmentAL[nrow].size();
	       for (int ni = 0; ni < nsize; ni++)
	       {
		   dtotscore += 1.0/nsize*
		       Util.distortion(clusterdata[nrow], 
                             currprofiles[((Integer) currassignmentAL[nrow].get(ni)).intValue()],pmavalues[nrow]);
	       }
	   }

	   if ((nrep == 0)||(dtotscore < dminscore))
	   {
	       bestcurrprofiles = currprofiles;
	       assignmentAL = currassignmentAL;
	       dminscore = dtotscore;
	   }
	}

	CenterRec[]  theCenterRecs = new CenterRec[bestcurrprofiles.length];
	for (int nindex = 0; nindex < theCenterRecs.length; nindex++)
	{
	    theCenterRecs[nindex] = new CenterRec(bestcurrprofiles[nindex], nindex);
	}
	Arrays.sort(theCenterRecs, new CenterRecCompare());
	int[] centerlookup = new int[theCenterRecs.length];
	modelprofiles = new double[currprofiles.length][];
	for (int nindex = 0; nindex < theCenterRecs.length; nindex++)
	{
	    modelprofiles[nindex] = theCenterRecs[nindex].center;
	    centerlookup[theCenterRecs[nindex].norigid] = nindex;
	}

  	bestassignments = new ArrayList[numrows];
        profilesAssigned = new ArrayList[modelprofiles.length];
	countassignments = new double[modelprofiles.length];
	expectedassignments = new double[modelprofiles.length];
	pvaluesassignments = new double[modelprofiles.length];
	significantnum = new boolean[modelprofiles.length];

        for (int nindex = 0; nindex < modelprofiles.length; nindex++)
	{
           profilesAssigned[nindex] =new ArrayList();
	   countassignments[nindex] = 0;
           expectedassignments[nindex] = -1;
	   pvaluesassignments[nindex] = 1;
	   significantnum[nindex] = false;
	}

        for (int nrow = 0; nrow < numrows; nrow++)
	{
	    bestassignments[nrow] = new ArrayList();
	    int nsize = assignmentAL[nrow].size();
	    for (int ni = 0; ni <nsize; ni++)
	    {
		int ncenter = centerlookup[((Integer)assignmentAL[nrow].get(ni)).intValue()];
		bestassignments[nrow].add(new Integer(ncenter));
	        profilesAssigned[ncenter].add(new Integer(nrow));
	        countassignments[ncenter]+=1.0/nsize;
	    }
	}
    }


    ///////////////////////////////////////////////////////////////////////////////////
    /**
     * Used for storing the coordinates of a k-means center and its ID
     */
   static class CenterRec
   {
       CenterRec(double[] center, int norigid)
       {
	   this.center = center;
	   this.norigid = norigid;
       }

       double[] center;
       int norigid;
   }

   ////////////////////////////////////////////////////////////////////////////
    /**
     * Used for comparing the records for two k-means centers
     */
   static class CenterRecCompare implements Comparator
   {
       public int compare(Object o1, Object o2)
       {
	   CenterRec c1 = (CenterRec) o1;
	   CenterRec c2 = (CenterRec) o2;

	   int nindex = 0;
	   while (nindex < c1.center.length)
	   {
	       if (c1.center[nindex] < c2.center[nindex])
	       {
		   return -1;
	       }
	       else if (c1.center[nindex] > c2.center[nindex])
	       {
		   return 1;
	       }
	       nindex++;
	   }
	   
           if (c1.norigid < c2.norigid)
           {
      	      return -1;
	   }
           else if (c1.norigid > c2.norigid)
	   {
	      return 1;
           }
           else
           {
      	      return 0;
	   }
       }
   }



    ////////////////////////////////////////////////////////////////
    /**
     * Generates the set of model profiles by first generating all candidate profiles
     */
    void generatemodelprofiles(int npatternlen, int nmaxchange)
    {
        //removes zero pattern after calling generatepatternsall
	if (npatternlen <= 1)
	{
	    modelprofiles = null;
	}

        else 
	{
           if (Math.pow((2*nmaxchange+1),npatternlen-1) < nsamplesmodel)
           {
              System.out.println("Generating all candidate model profiles");
	      double[][] patterns = generatemodelprofilesall(npatternlen, nmaxchange);
              modelprofiles = new double[patterns.length-1][npatternlen];

              for (int nrow = 0; nrow < modelprofiles.length/2; nrow++)
      	      {
	         for (int ncol = 0; ncol < npatternlen; ncol++)
	         {
                    modelprofiles[nrow][ncol] = patterns[nrow][ncol]; 
	         } 
	      }

              for (int nrow = modelprofiles.length/2+1; nrow < patterns.length; nrow++)
  	      {
	         for (int ncol = 0; ncol < npatternlen; ncol++)
	         {
                    modelprofiles[nrow-1][ncol] = patterns[nrow][ncol]; 
	         }
	      }
           }
           else
           {
              System.out.println("Sampling Candidate Model Profiles");
              modelprofiles = generatemodelsampled(npatternlen,nmaxchange);
           }

           compactprofiles2(dmaxcorrmodel,nmaxprofiles);
	}
    }

    /////////////////////////////////////////////////////////////////
    /**
     * Generates the set of model profiles by first sampling a set of model profiles
     */
    double[][] generatemodelsampled(int npatternlen, int nmaxchange)
    {

        double[][] modelsamples = new double[(int) nsamplesmodel][npatternlen];
        Random theRandom = new Random(3733246);    

        modelsamples[0][0] = 0;
        for (int nindex = 1; nindex < npatternlen; nindex++)
        {
           modelsamples[0][nindex] = modelsamples[0][nindex-1]-nmaxchange; 
        }

        for (int nsample = 1; nsample < nsamplesmodel; nsample++)
        {
           modelsamples[nsample][0] = 0;
           for (int nindex = 1; nindex < npatternlen; nindex++)
           {
              modelsamples[nsample][nindex] = modelsamples[nsample][nindex-1]+
                     Math.floor(theRandom.nextDouble()*(2*nmaxchange+1)-nmaxchange);
           }
        }

        return modelsamples;
    }

    

    ////////////////////////////////////////////////////////
    /**
     * Returns true iff neighborProfile correlation with all profiles in tsNeighborhood
     * is greater than or equal to dminclustdist
     */
    public boolean closeToAllNeighbors(ProfileRec insertProfile, TreeSet tsNeighborhood)
    {



        Iterator itrtsNeighborhood = tsNeighborhood.iterator();
        ProfileRec otherProfile;
        boolean bclose = true;

        while ((itrtsNeighborhood.hasNext())&&(bclose))
	{
            otherProfile = ((ProfileRecDist) itrtsNeighborhood.next()).theProfileRec;
            if (Util.correlation(modelprofiles[insertProfile.nprofileindex],
                           modelprofiles[otherProfile.nprofileindex])<dmaxminclustdist)
	    {
		bclose = false;
	    }
	}
        return bclose;
    }

    ///////////////////////////////////////////////////////////////////////////

    /**
     * Method for clustering profiles
     */
    public void clusterprofiles(boolean[] significant, ArrayList clustersofprofiles,
				boolean bnormal) throws Exception
    {
        dmaxminclustdist = dminclustdist;
        NumberFormat nf2 = NumberFormat.getInstance(Locale.ENGLISH);
        nf2.setMinimumFractionDigits(2);
        nf2.setMaximumFractionDigits(2);

        if ((sortedcorrvals != null)&&(dpercentileclust >= 0))
        {			
            int nindex =(int) Math.min(sortedcorrvals.length - 1,Math.floor(dpercentileclust*sortedcorrvals.length));
            System.out.println("Minimum correlation for clustering based on percentile: "+nf2.format(sortedcorrvals[nindex]));
			
            if (sortedcorrvals[nindex] > dmaxminclustdist)
	       dmaxminclustdist = sortedcorrvals[nindex];		
        }		
        int numprofiles = modelprofiles.length;

	//count significant and non-significant
        int numsignificant = 0;
	for (int nindex = 0; nindex < numprofiles; nindex++)
	{
	    if (significant[nindex]==bnormal)
	    {
                numsignificant++;
	    }
	}

        ArrayList significantprofiles = new ArrayList();
        ArrayList bestCluster;

        int nsignificantindex = 0;

        //putting all profiles into either significant or nonsignificant sets 
        for (int nindex = 0; nindex < numprofiles; nindex++)
	{  
            ProfileRec newProfileRec = new ProfileRec(nindex, countassignments[nindex], 
                                      expectedassignments[nindex], pvaluesassignments[nindex]);
	    if (significant[nindex]==bnormal)
	    {
                significantprofiles.add(newProfileRec);
	    }
	}

        int numsignificantprofiles;

	while ((numsignificantprofiles = significantprofiles.size())>0)
        {
            double dbestnumgenes = 0;              
            TreeSet bestBall = new TreeSet();
          
            for (int nprofileindex = 0; nprofileindex < numsignificantprofiles; nprofileindex++)
	    {
                //finding the best profile to grow a ball around
                ProfileRec centerProfile = (ProfileRec) significantprofiles.get(nprofileindex);
                double dnumgenessig = centerProfile.dnumgenes;

                TreeSet tsneighborProfileCandidates = new TreeSet(new Profilerecdistcomparator());
		//should sort significant on something              
                for (int notherindex = 0; notherindex < numsignificantprofiles; notherindex++)
	        {
                    if (notherindex != nprofileindex)
		    {
			ProfileRec neighborProfile = (ProfileRec) significantprofiles.get(notherindex);
                        double dtempcorrval = Util.correlation(modelprofiles[centerProfile.nprofileindex],
                                                modelprofiles[neighborProfile.nprofileindex]);

			if (dtempcorrval>dmaxminclustdist)
			{
			   tsneighborProfileCandidates.add(
                                            new ProfileRecDist(neighborProfile,dtempcorrval));
			}
		    }
		}

    
                TreeSet tsneighborProfile = new TreeSet(new Profilerecdistcomparator());
                Iterator tsneighborCandidateIterator = tsneighborProfileCandidates.iterator();
               
                ProfileRecDist currProfileRecDist;
                ProfileRec currProfileRec;
                //profile is a candidate profile if within a bound distance of center profile
                //add profiles to cluster by order and if near all other profiles already added
                
                while (tsneighborCandidateIterator.hasNext())
		{
		    currProfileRecDist = (ProfileRecDist) tsneighborCandidateIterator.next();    
                    currProfileRec = currProfileRecDist.theProfileRec;
                    if (closeToAllNeighbors(currProfileRec, tsneighborProfile))
		    {  
		        tsneighborProfile.add(currProfileRecDist);
                        dnumgenessig += currProfileRec.dnumgenes;
		    }
		}               

               if (dnumgenessig > dbestnumgenes)
                {
                    dbestnumgenes = dnumgenessig;
                    tsneighborProfile.add(new ProfileRecDist(centerProfile,1));
                    bestBall = tsneighborProfile;
                }
             } //for 

            //remove from significant set those profiles in bestcluster
            Iterator trBestBall = bestBall.iterator();
            bestCluster = new ArrayList();
            while (trBestBall.hasNext())
            {
                
		boolean bfound = false;
                int nclustersigindex = 0;
                ProfileRecDist bestRecDist = (ProfileRecDist) trBestBall.next();
                ProfileRec bestRec = bestRecDist.theProfileRec;
                bestCluster.add(bestRec);
                while ((!bfound)&&(nclustersigindex < significantprofiles.size()))
		{
                    ProfileRec sigRec = (ProfileRec) significantprofiles.get(nclustersigindex);
                    if (bestRec.nprofileindex == sigRec.nprofileindex)
		    { 
			significantprofiles.remove(nclustersigindex);
                        bfound = true;
		    }
                    nclustersigindex++;	
                 }               
	    }
            clustersofprofiles.add(bestCluster);
         } // while
    }


    ////////////////////////////////////////////////////////////////
    /**
     * Exhaustively generates all possible model profiles
     */
    static double[][] generatemodelprofilesall(int npatternlen, int nmaxchange)
    {
        if (npatternlen == 1)
	{
	    return new double[][] {{0}};
	}
        else
	{
	    double[][] patterns = generatemodelprofilesall(npatternlen-1,nmaxchange);
            int nchoices = 2*nmaxchange+1;
            double[][] newpatterns = new double[patterns.length*nchoices][npatternlen];

            for (int nrow = 0; nrow < newpatterns.length; nrow++)
	    {
		for (int ncol = 0; ncol < npatternlen-1; ncol++)
		{
		    newpatterns[nrow][ncol] = patterns[nrow/nchoices][ncol];
		}

                newpatterns[nrow][npatternlen-1] = 
                         patterns[nrow/nchoices][npatternlen-2] + (nrow % nchoices) - nmaxchange;
	    }
          
            return newpatterns;
	}
    }



   //////////////////////////////////////////////////////////////////////
    /**
     * Computes the expected number of genes assigned to a profile based on a permutation test
     * of time points
     */
    public void computeaveragetally()
    {
	//computes numbered assigned for each permutation
	int[][] permutations = null;
        //generate all perumtations in memory if all permutations requested or not too small
        boolean bgenallperms =((numcols <ALLPERMSTHRESH)||(nsamplesgene <= 0));

        expectedassignments = new double[modelprofiles.length];
        for (int nindex = 0; nindex < expectedassignments.length; nindex++)
	{
	   expectedassignments[nindex] = 0;
        }        

        int ntotalassignments = 0;

        double davgx   = 0;
        double dsumxsq = 0;
        double dsqrtx  = 0;
        int numx       =0;
         
        double[] dsumy = new double[modelprofiles.length];
        double[] dsumysq = new double[modelprofiles.length];
        double[] dsqrty = new double[modelprofiles.length];

        modelprofilestats(dsumy,dsumysq,dsqrty);

        int nbegin=-1;
       
        int[] permutedpmavalues = new int[numcols];
        double[] permutedlogdata = new double[numcols];
        double[] logdata = new double[numcols];
        int[] bestassignments = new int[modelprofiles.length];

        int nselectedperms;
        int[] selectedperms;
        boolean ballperms;  //this is if all permutation should be generated, except possibly the 0 point
	//if ballpermutevals is also true then all permutations will be generated for that case.

        double[] vals = new double[4];
	double[] repeatvals = null;
	int numrepeats = 0;

	if (generepeatspottimedata != null)
	{
            //we got to re-merge the data sets after renormalizing
            numrepeats = generepeatspottimedata[0].length;
	    repeatvals = new double[numrepeats];
	}

        if (bgenallperms)
        {
	   //explicitly generating all permutations - though may not use them all
	   //need to include other way also
	   if (ballpermuteval)
	   {
              permutations = generatepermutations(numcols);
           }
           else
	   {
	      permutations = generatepermutationsExcept0(numcols);  //try other perm method
	   }

           nselectedperms = Math.min(nsamplesgene, permutations.length);
	   //if more samples requested than perumutation then just do all permutations   
           if ((nselectedperms <= 0)||(nselectedperms == permutations.length))
           {
	      //we are going to do all permutations, mark each selected permutation as an actual
              selectedperms = new int[permutations.length];
              for (int nindex = 0; nindex < selectedperms.length; nindex++)
              {
                 selectedperms[nindex] = nindex;
              } 
              ballperms = true;
           }
           else
           {
	       //we are going to select the subset of permutations 
              ballperms = false;
              selectedperms = new int[nselectedperms];
           }
        }
        else
        {
           ballperms = false;
           selectedperms = new int[nsamplesgene];

           //going to generate desired permutation
           for (int nindex = 0; nindex < selectedperms.length; nindex++)
           {
              selectedperms[nindex] = nindex;
           }
        }

        Random theRandom = new Random(9873287);

	//pulled this out of the inner loop to save memory, but only need for inner loop
	boolean[] picked=null;
	int npermstart = 0;
	if ((!ballperms)&&(!bgenallperms))
	{
           permutations = new int[selectedperms.length][numcols];

	   if (!ballpermuteval)
	   {
	       npermstart = 1;
	   }

           picked = new boolean[numcols-npermstart];
	}

        for (int nrow = 0; nrow < numrows; nrow++)
        {
	    nbegin = -1;
	   if (nrow % 100 == 0)
	   {
              System.out.print(".");
	   }
     
           int[] currpmavalues = pmavalues[nrow];

           if (!ballperms)
           {
              if (bgenallperms)
              {
		  //explicitly generated all permutations
                 for (int nindex = 0; nindex < selectedperms.length; nindex++)
                 {
                    selectedperms[nindex] = 
                         (int) Math.floor((theRandom.nextDouble()*permutations.length));
                 }
                 Arrays.sort(selectedperms);
              }
              else
              {
                 for (int nindex = 0; nindex < selectedperms.length; nindex++)
                 {
                     for (int njindex = 0; njindex < numcols-npermstart; njindex++)
                     {
                        picked[njindex] = false;
                     }

                     for (int njindex = npermstart; njindex < numcols; njindex++)
                     {
			 //randomly generating a permutation 
                        int nmove = (int) Math.floor((numcols-njindex)*theRandom.nextDouble());
                        //nmove is the element to select
                        int nkindex = 0;
                        int ncount = 0;
                        while ((ncount < nmove)||(picked[nkindex]))
                        {
                           if (!picked[nkindex])
                           {
                              ncount++;
                           }
                           nkindex++;
                        }
                        picked[nkindex]  = true;
                        permutations[nindex][njindex] = nkindex+npermstart;
                     }
                 }
              }
           }


           //got the permutations
  	   for (int npermutationnum = 0; npermutationnum <selectedperms.length; npermutationnum++)
	   {
               int[] currperm = permutations[selectedperms[npermutationnum]];

               if (currpmavalues[currperm[0]]!=0)
               {
                  //legal time point 0
                  if (nbegin != currperm[0])
                  {
		      //going to need to renormalize changing first time point
                     nbegin = currperm[0];

                     //assuming data is already in log space
		     if (generepeatspottimedata == null)
		     {
			 //no repeats just the single data set

		        double[][] spottimedata = genespottimedata[nrow];
                        int[][] spottimepma = genespottimepma[nrow];

			///////////double growth		     
                        if (spottimepma.length > vals.length)
			{
			    vals = new double[2*spottimepma.length];
			}

                        for (int ncol = 0; ncol < numcols; ncol++)
			{
			   int nvalindex = 0;
                           for (int nspot = 0; nspot < spottimepma.length; nspot++)
  			   {
			       //iterating over all source spots for row
  		              if (spottimepma[nspot][ncol]!=0)
	                      {
                                 vals[nvalindex] = spottimedata[nspot][ncol]-spottimedata[nspot][nbegin];
	                         nvalindex++;
         	              }
			   }
		      
                           if (nvalindex > 0)
			   {
                              logdata[ncol] = Util.getmedian(vals,nvalindex);
			   }
			}
		     }
		     else
		     {  
                        for (int ncol = 0; ncol < numcols; ncol++)
			{
		           //we got to iterate over each repeat also
			   int nrepeatvalsindex = 0;
			   for (int nrepeat = 0; nrepeat < numrepeats; nrepeat++)
		           {
			       //iterating over all repeats
		              double[][] spottimedata = generepeatspottimedata[nrow][nrepeat];
                              int[][] spottimepma = generepeatspottimepma[nrow][nrepeat];
	     
                              if (spottimepma.length > vals.length)
			      {
  			         vals = new double[2*spottimepma.length];
			      }
     
			      int nvalindex = 0;
                              for (int nspot = 0; nspot < spottimepma.length; nspot++)
  			      {
                                  //iterating over all source spots
  		                 if (spottimepma[nspot][ncol]!=0)
	                         {
                                    vals[nvalindex] = spottimedata[nspot][ncol]-spottimedata[nspot][nbegin];
	                            nvalindex++;
         	                 }
			      }
			   
                              if (nvalindex > 0)
			      {
                                 repeatvals[nrepeatvalsindex] = Util.getmedian(vals,nvalindex); 
                                 nrepeatvalsindex++;
			      }
			   }

                           if (nrepeatvalsindex > 0)
			   {
			      logdata[ncol] = Util.getmedian(repeatvals,nrepeatvalsindex);
			   }
			}
		     }

                     
                     davgx = 0;
                     dsumxsq = 0;
                     numx = 0;
                     for (int ncol = 0; ncol < numcols; ncol++)
                     {
			if (permutedpmavalues[ncol] != 0)
                        {
                           numx++;
                           davgx += logdata[ncol];
                           dsumxsq  += logdata[ncol]*logdata[ncol];
                        }
                     }
                     dsqrtx = Math.sqrt(dsumxsq - davgx*davgx/numx);
                     davgx = davgx/numx;
                  }
                
               
                  int[] currmpv = pmavalues[nrow];
                  for (int ncol = 0; ncol < numcols; ncol++)
                  {
                     permutedlogdata[ncol] = logdata[currperm[ncol]];
                     permutedpmavalues[ncol] = currmpv[currperm[ncol]];
                  }

                  int numbest = 0;
                  double dcorr;
                  double dcorrmax = -2;
            
                  for (int nmodelprofile = 0; nmodelprofile < modelprofiles.length; nmodelprofile++)
  	          {
                     double[] currprofile = modelprofiles[nmodelprofile];


                     if (numx == numcols)
                     {
 	                double dsumxy = 0;
                        for (int nindex = 0; nindex < numcols; nindex++)
	                {
                           dsumxy  += permutedlogdata[nindex]*currprofile[nindex];
	                }

                        dcorr = (dsumxy - davgx*dsumy[nmodelprofile])/(dsqrtx*dsqrty[nmodelprofile]);
                        
		     }
                     else
                     {
		         dcorr = Util.correlation(permutedlogdata,
                                            modelprofiles[nmodelprofile],
                                            permutedpmavalues);
                     }

                     if (dcorr == dcorrmax)
  		     {
                         bestassignments[numbest] = nmodelprofile;
                         numbest++;
                     }	    
                     else if (dcorr > dcorrmax)
		     {
                         bestassignments[0] = nmodelprofile;
                         numbest = 1;
                         dcorrmax = dcorr;
		     }
		      //if dcorr na then numbest not advanced
                  } //model profile

                  if (numbest > 0)
		  {
		     ntotalassignments++;
                     double dweight = 1.0/(double) numbest;

                     for (int nindex = 0; nindex < numbest; nindex++)
                     {
	                expectedassignments[bestassignments[nindex]] += dweight;
	             }
		  }
               }//valid
	   } //permutation                     
	} //row

        System.out.println();
        for (int nindex = 0; nindex < expectedassignments.length; nindex++)
	{
	   expectedassignments[nindex] = expectedassignments[nindex]*(double)numrows/(double) ntotalassignments;
	}
    } 

    ////////////////////////////////////////////////////////////////////////////////
    /**
     * Sets all genes to be assigned to profile index 0
     */
    public void assignall0()
    {
        profilesAssigned = new ArrayList[1];

	bestassignments = new ArrayList[numrows];
	countassignments = new double[numrows];
        ArrayList al0 = new ArrayList();
	al0.add(new Integer(0));
	profilesAssigned[0] = new ArrayList();
        for (int nrow = 0; nrow < numrows; nrow++)
	{
	    bestassignments[nrow] = al0;
	    countassignments[nrow] = 0;
	    profilesAssigned[0].add(new Integer(nrow));
	}
    }


    //////////////////////////////////////////////////////////////////////////////////
    /**
     * Generates all permutations that do not involve permuting the 0 time point
     */
    int[][] generatepermutationsExcept0(int numelements)
    {
        numelements--;
        int[] elements = new int[numelements];
        int nfact = 1;
        for (int nindex = 2; nindex <= numelements; nindex++)
	{
	    nfact *= nindex;
	}

        int[][] permutations = new int[nfact][numelements+1];

        for (int nindex = 0; nindex < elements.length; nindex++)
	{
            elements[nindex] = nindex+1;
            permutations[0][nindex+1] = nindex+1;
	}

	for (int nperm = 0; nperm < permutations.length; nperm++)
	{
	    permutations[nperm][0] = 0;
	}

        //algorithm from  Knuth 7.2.1.2 
        int j = numelements-2;
        int k,l;
        int npermutationnum = 0;
        while (j>=0)
        {
	    for (int nindex = 0; nindex < numelements; nindex++)
		 permutations[npermutationnum][nindex+1]= elements[nindex];
            npermutationnum++;
	    
            j = numelements - 2;
	    while ((j>=0) && (elements[j] >= elements[j+1]))
	      j--;

	    if (j >= 0)
	    {
               l = numelements-1;
              while (elements[j]>=elements[l])
	      {
	         l--;
	      }
              swap(elements,j,l);
              k = j + 1;
              l = numelements-1;

              while (k < l)
	      {
	         swap(elements, k, l);
	         k++;
                 l--;
	      }
	   }
	}

        return permutations;
    }


    //////////////////////////////////////////////////////////////////////////////////
    /**
     * Generates all permutations including those that permute the 0 time point
     */
    int[][] generatepermutations(int numelements)
    {
       
        int[] elements = new int[numelements];
        int nfact = 1;
        for (int nindex = 2; nindex <= numelements; nindex++)
	{
	    nfact *= nindex;
	}

        int[][] permutations = new int[nfact][numelements];

        for (int nindex = 0; nindex < elements.length; nindex++)
	{
            elements[nindex] = nindex;
            permutations[0][nindex] = nindex;
	}

        //algorithm from  Knuth 7.2.1.2 
        int j = numelements-2;
        int k,l;
        int npermutationnum = 0;
        while (j>=0)
        {
	    for (int nindex = 0; nindex < numelements; nindex++)
		 permutations[npermutationnum][nindex]= elements[nindex];
            npermutationnum++;
	    
            j = numelements - 2;
	    while ((j>=0) && (elements[j] >= elements[j+1]))
	      j--;

	    if (j >= 0)
	    {
               l = numelements-1;
              while (elements[j]>=elements[l])
	      {
	         l--;
	      }
              swap(elements,j,l);
              k = j + 1;
              l = numelements-1;

              while (k < l)
	      {
	         swap(elements, k, l);
	         k++;
                 l--;
	      }
	   }
	}

        return permutations;
    }
   
    //////////////////////////////////////////////////////////////////////////
    /**
     * Swap els[a] and els[b]
     */
    private void swap(int[] els, int a, int b)
    {
       int t = els[a];
       els[a] = els[b];
       els[b] = t;
    }

    ////////////////////////////////////////////////////////////////////////////////
    /**
     * Computes sum and variance statistics on model profiles
     */
    public void modelprofilestats(double[] dsumy, double[] dsumysq, double[] dsqrty)
    {
       
        for (int nrow = 0; nrow < modelprofiles.length; nrow++)
        {
           dsumy[nrow] = 0;
           dsumysq[nrow] = 0;
           for (int ncol = 0; ncol < numcols; ncol++)
           {
             dsumy[nrow] += modelprofiles[nrow][ncol];
             dsumysq[nrow] += modelprofiles[nrow][ncol]*
                              modelprofiles[nrow][ncol];
           }
           dsqrty[nrow] = Math.sqrt(dsumysq[nrow] - 
                                   dsumy[nrow]*dsumy[nrow]/numcols);
        }
    }    

    /////////////////////////////////////////////////////////////////////////////////
    /**
     * Implements the greedy algorithm to select a subset of the candidate model profiles
     * based on dmaxcorrelationprofiles and nmaxprofiles
     */
    public void compactprofiles2(double dmaxcorrelationprofiles, 
                                 int nmaxprofiles)
    {
        boolean[]  selected = new boolean[modelprofiles.length];

        double[] dsumy = new double[modelprofiles.length];
        double[] dsumysq = new double[modelprofiles.length];
        double[] dsqrty = new double[modelprofiles.length];

        modelprofilestats(dsumy, dsumysq, dsqrty);
              
        double dminval;
        int nminindex;  
        double dstopval = Math.min(dmaxcorrelationprofiles,1.5);
        int nstopnum = Math.min(modelprofiles.length, nmaxprofiles);
        if (nstopnum <= 0) 
	{
           nstopnum = modelprofiles.length;
	}

        int[] selectedindex = new int[nstopnum];
        double r =(double)(1)/(double) (2*nmaxchange+1);
        double rsum = (((double)
			(Math.pow(r,modelprofiles[0].length)-1))/((double)(r-1))-1);

        int ndown = (int)(modelprofiles.length*(nmaxchange+1)*rsum);
        selected[ndown] = true;

        int nsetsize = 1;

        if (nstopnum > 1)
	{
            selectedindex[0] = ndown;
           double[] closestcorr = new double[modelprofiles.length];
           double[] currmodelprofile1 = modelprofiles[ndown];
           double[] currmodelprofile2;
           double dsumxy;
           double ddistnew;
  
           for (int nindex = 0; nindex < closestcorr.length; nindex++)
           {
              currmodelprofile2 = modelprofiles[nindex];
              dsumxy = 0;
              for (int ninnerindex = 0; ninnerindex < numcols; ninnerindex++)
	      {
                 dsumxy  += currmodelprofile1[ninnerindex]*currmodelprofile2[ninnerindex];
	      }
              closestcorr[nindex] = (dsumxy - dsumy[nindex]*dsumy[ndown]/numcols)
                                           /(dsqrty[nindex]*dsqrty[ndown]);
           }
 
           double[] maxrawval = new double[modelprofiles.length];

           for (int nprofile = 0; nprofile < maxrawval.length; nprofile++)
	   {
	       maxrawval[nprofile] = modelprofiles[nprofile][0];

               for (int nrawindex = 1; nrawindex < modelprofiles[nprofile].length; nrawindex++)
	       {
                   double dtempval =Math.abs(modelprofiles[nprofile][nrawindex]); 
		   if (dtempval>maxrawval[nprofile])
		   {
		       maxrawval[nprofile] = dtempval;
		   }
	       } 
	   }

           do 
	   {
              nminindex = -1;
              dminval = 2; //correlation
	      double dminmaxabs = nmaxchange *modelprofiles[0].length;
              for (int nindex = 0; nindex < modelprofiles.length; nindex++)
	      {
                  if (!selected[nindex])
		  {		

                      boolean btry =true;
		      if  ((Math.abs(closestcorr[nindex]-dminval)<=FLOATERROR)&&
			   (maxrawval[nindex] < dminmaxabs))
 		      {

                        dsumxy = 0;
                        for (int ninnerindex = 0; ninnerindex < numcols; ninnerindex++)
	                {
                           dsumxy  += modelprofiles[nminindex][ninnerindex]
                                        *modelprofiles[nindex][ninnerindex];
	                }
                                 
                        double dcorrval = (dsumxy - dsumy[nindex]*dsumy[nminindex]/numcols)
                                        /(dsqrty[nindex]*dsqrty[nminindex]);
                    
                        if (dcorrval+FLOATERROR >= 1)
			{
			   dminval = closestcorr[nindex];
                           nminindex = nindex;
                           dminmaxabs = maxrawval[nindex];
                           btry = false;
	        	}
		     }

		     if ((btry) &&(closestcorr[nindex] < dminval))
		     {
                        dminval = closestcorr[nindex];
                        nminindex = nindex;
                        dminmaxabs = maxrawval[nindex];
	             }
	          }
	       }

               if ((nminindex > -1)&&(dminval+FLOATERROR < dmaxcorrelationprofiles))
	       {
                 selected[nminindex] = true;
                 selectedindex[nsetsize] = nminindex;
                 nsetsize++;           
                 currmodelprofile1 = modelprofiles[nminindex];   
                 for (int npindex = 0; npindex < closestcorr.length; npindex++)
                 {
                    currmodelprofile2 = modelprofiles[npindex];
                    dsumxy = 0;
                    for (int ninnerindex = 0; ninnerindex < numcols; ninnerindex++)
	            {
                       dsumxy  += currmodelprofile1[ninnerindex]*currmodelprofile2[ninnerindex];
	            }

                    ddistnew  = (dsumxy - dsumy[nminindex]*dsumy[npindex]/numcols)
                                         /(dsqrty[nminindex]*dsqrty[npindex]);

                    if (ddistnew > closestcorr[npindex])
		    {
                      closestcorr[npindex]= ddistnew;
		    }
                 }
	      }
	   } while ((dminval+FLOATERROR < dstopval)&&(nsetsize<nstopnum));
	
	   
           NumberFormat nf3 = NumberFormat.getInstance(Locale.ENGLISH);
           nf3.setMinimumFractionDigits(3);
           nf3.setMaximumFractionDigits(3);
           System.out.println("Maximum correlation between model profiles: "+nf3.format(dminval));
	}
        double[][] tempmodelprofiles = new double[nsetsize][numcols];
       
        Arrays.sort(selectedindex, 0, nsetsize);
        for (int nindex = 0; nindex < nsetsize; nindex++)
	{
           tempmodelprofiles[nindex] = modelprofiles[selectedindex[nindex]];
	}
        modelprofiles = tempmodelprofiles;
    }


    ////////////////////////////////////////////////////////////////////////////////////////////
    /**
     * Finds the best profile which matches each gene
     */
    public void findbestgroupassignments()
    {         
        profilesAssigned = new ArrayList[modelprofiles.length];
      
        ArrayList allprofiles = new ArrayList();
        for (int nindex = 0; nindex < modelprofiles.length; nindex++)
	{
           allprofiles.add(new Integer(nindex));
           profilesAssigned[nindex] =new ArrayList();
	}
       
	bestassignments = new ArrayList[numrows];
        double dcorr;

        for (int nrow = 0; nrow < numrows; nrow++)
	{
            double dcorrmax = -2;
            
            bestassignments[nrow] = allprofiles;
            for (int nmodelprofile = 0; nmodelprofile < modelprofiles.length; nmodelprofile++)
	    {
		dcorr = Util.correlation(data[nrow],modelprofiles[nmodelprofile],
                                         pmavalues[nrow]);

                if (dcorr == dcorrmax)
		{
		    bestassignments[nrow].add(new Integer(nmodelprofile));
                }	    
                else if (dcorr > dcorrmax)
		{
                    dcorrmax = dcorr;
                    bestassignments[nrow] = new ArrayList();
                    bestassignments[nrow].add(new Integer(nmodelprofile));
		}
            }
            for (int nindex = 0; nindex < bestassignments[nrow].size(); nindex++)
            {
               profilesAssigned[((Integer) bestassignments[nrow].get(nindex)).intValue()].add(new Integer(nrow));
            }
	}       
    }

}