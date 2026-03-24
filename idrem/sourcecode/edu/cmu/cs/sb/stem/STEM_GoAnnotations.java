package edu.cmu.cs.sb.stem;
import edu.cmu.cs.sb.core.*;
import java.util.*;
import java.io.*;

/**
 * Extends GoAnnotations with methods specific to STEM for analyzing GoAnnotations
 */
public class STEM_GoAnnotations extends GoAnnotations
{

    /**
     * Consructor just calls parent constructor
     */
    public STEM_GoAnnotations(String szorganismsourceval,String szxrefsourceval,
                  String szxrefval, String szGoFile,String szGoCategoryFile, 
                  String[] baseGenes1, String[] baseProbes1, int nsamplespval,
                  int nmingo,int nmingolevel,String szextraval,String szcategoryIDval,
                  boolean bspotincluded, String szevidenceval,String sztaxonval,
			      boolean bpontoval,boolean bcontoval,boolean bfontoval,boolean brandomgoval,String szBatchGOoutput) 
                        throws FileNotFoundException, IOException,IllegalArgumentException
    {
	super(szorganismsourceval, szxrefsourceval,
              szxrefval, szGoFile, szGoCategoryFile, 
              baseGenes1, baseProbes1, nsamplespval,
              nmingo, nmingolevel, szextraval, szcategoryIDval,
              bspotincluded, szevidenceval, sztaxonval,
	      bpontoval, bcontoval, bfontoval, brandomgoval,szBatchGOoutput);
    }

    /////////////////////////////////////////////////////////////////////////////////////
    /**
     *Used to get the GO results for a cluster of profiles
     *the id of each profile in the cluster is listed in profileList
     *an array list in ids has all the indicies in genenames
     *assigned to the profile
     *If bqueryset is true, then only genes also in the query set will be used
     *If htinames is non-null, then only genes also in htinames will be used
     */
    public GoResults getCategory(ArrayList profileList, ArrayList[] ids, 
				 String[] genenames, ArrayList[] assignments, 
                                 boolean bqueryset,HashSet htinames)
    {

       ArrayList combinedIDs = new ArrayList();
       int npls = profileList.size();
       for (int nprofile = 0; nprofile < npls; nprofile++)
       {
	   //iterating through ProfileRec
          STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) profileList.get(nprofile);
	  combinedIDs.addAll(ids[pr.nprofileindex]);
       }
       
       return getCategory(combinedIDs, genenames, assignments,bqueryset,htinames,-1);      
    }
   

    ////////////////////////////////////////////////////////////////////////////////////
    /**
     * Puts in tpgr a sorted ranking of profiles based on the current sort by seelctions
     */
    public void getProfileRankings(ArrayList clustersofprofilesnum,
                                   ArrayList[] profilesAssigned, String[] genenames,
			           ArrayList[] assignments,double[] expectedassignments)
    {

       double dweight;
       tpgr = new ProfileGORankingRec[profilesAssigned.length];

       boolean breallyquery = (szsortcommand.equals("define")|| szsortcommand.equals("expdefine"));
      
       int ncategoryall;
       if (breallyquery)
       {
          ncategoryall = nGeneSet; 
       }
       else
       {        
          ncategoryall = ((Integer) htFullCount.get(szSelectedGO)).intValue();
       }

       double dcategoryselect;
       double dmaxselect;

       int numclusters = clustersofprofilesnum.size();

       //first compute the GO enrichment for profiles
       for (int nprofile = 0; nprofile < profilesAssigned.length; nprofile++)
       { 
          RecCount theRecCount = incrementSelectData(profilesAssigned[nprofile],
                                      genenames,assignments,breallyquery,szSelectedGO);
          dcategoryselect  = theRecCount.dmatch;
          dmaxselect  = theRecCount.dtotal;
          double dpval;

	  if ((szsortcommand.equals("expgo"))||(szsortcommand.equals("expdefine")))
	  {
             dpval = StatUtil.binomialtail((int) Math.ceil(dcategoryselect-1), ncategoryall, 
                                     expectedassignments[nprofile]/numtotalgenes);
	  }
          else
	  {
             dpval = StatUtil.hypergeometrictail((int) Math.ceil(dcategoryselect-1), ncategoryall, 
                                     numtotalgenes-ncategoryall,(int) Math.ceil(dmaxselect));
	  }

          tpgr[nprofile] =  new ProfileGORankingRec(nprofile,numclusters,
						    dpval,2,dcategoryselect,dmaxselect,
                                                    0,0,ncategoryall, numtotalgenes);
       }

       if (bcluster)
       {
	  double dclusterpval; 

          for (int ncluster = 0; ncluster < numclusters; ncluster++)
          {
             double dcategoryselectcluster = 0;
             double dmaxselectcluster = 0;

             ArrayList currprofiles = (ArrayList) clustersofprofilesnum.get(ncluster);
             int nsize = currprofiles.size();
             for (int nprofileindex = 0; nprofileindex < nsize; nprofileindex++)
	     {
	        STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) currprofiles.get(nprofileindex);
                dcategoryselectcluster += tpgr[pr.nprofileindex].dgenes;
                dmaxselectcluster += tpgr[pr.nprofileindex].dmaxselect;
	     } 

             dclusterpval  = StatUtil.hypergeometrictail((int) Math.ceil(dcategoryselectcluster-1), 
                                         ncategoryall, numtotalgenes-ncategoryall,(int) Math.ceil(dmaxselectcluster));
	  
             for (int nprofileindex = 0; nprofileindex < nsize; nprofileindex++)
	     {
	        STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) currprofiles.get(nprofileindex);
                tpgr[pr.nprofileindex].dclusterpval = dclusterpval;
                tpgr[pr.nprofileindex].dgenescluster = dcategoryselectcluster;
                tpgr[pr.nprofileindex].dmaxselectcluster = dmaxselectcluster;             
                tpgr[pr.nprofileindex].ncluster = ncluster;
             } 
          }
       } 
       //sorts the ranking records
       Arrays.sort(tpgr, new GORankingComparator()); 
    }


    //////////////////////////////////////////////////////////////////////////
    /**
     * Loads theRecIDpval and theClusterRecIDpval to have the p-values for
     * the most enriched profiles and clusters for each category
     */
    public void computeBestPvaluesClustersProfiles(STEM_DataSet theDataSet)
    {
       HashMap hIDpval = new HashMap();  //maps ID to best p-val
       HashMap hIDpvalProfile = new HashMap(); //maps ID to best p-vals for hyper and binomial
      
       double dweight;

       String[] genesprobes = theDataSet.genenames;

       int numclusters =  theDataSet.clustersofprofilesnum.size();

       //keeps track if already analyzed a profile while analyzing clusters of profiles
       boolean[] banalyzed = new boolean[theDataSet.profilesAssigned.length];
       for (int nindex = 0; nindex < banalyzed.length; nindex++)
       {
	   banalyzed[nindex] = false;
       }

       for (int ncluster = 0; ncluster < numclusters; ncluster++)
       {      
	  double dselect = 0;
          ArrayList currcluster = (ArrayList) theDataSet.clustersofprofilesnum.get(ncluster);
          int nclustersize = currcluster.size();
          HashMap htGoCounts = new HashMap();
	  for (int nprofileindex = 0; nprofileindex < nclustersize;  nprofileindex++)
	  {
	     HashMap htGoCountsProfile = new HashMap();

	     STEM_DataSet.ProfileRec pr = (STEM_DataSet.ProfileRec) currcluster.get(nprofileindex); 
	     int nprofile = pr.nprofileindex;
             double dselectprofile = theDataSet.countassignments[nprofile];
	     dselect += dselectprofile;
             banalyzed[nprofile] = true; 
             int npasize = theDataSet.profilesAssigned[nprofile].size();
             for (int nindex = 0; nindex < npasize; nindex++)
	     {
                int ngeneindex = ((Integer) theDataSet.profilesAssigned[nprofile].get(nindex)).intValue();
                String szGene =  genesprobes[ngeneindex];

                HashSet goList = labelsForID(szGene);
		Iterator goListitr = goList.iterator();
              
		double davgweight = 1/(double) theDataSet.bestassignments[ngeneindex].size();
		   
	        while (goListitr.hasNext())
	        {
		   String szgoid = (String) goListitr.next();
		   Double dcount = (Double) htGoCounts.get(szgoid);
                   if (dcount == null)
		   {
	              htGoCounts.put(szgoid, new Double(davgweight));
                      //if not in cluster counts then we know it is also not in profile 
	              htGoCountsProfile.put(szgoid, new Double(davgweight));
	           }
                   else
                   {
           	      htGoCounts.put(szgoid, new Double(dcount.doubleValue()+davgweight));
		      Double dcountprofile = (Double) htGoCountsProfile.get(szgoid);
                      if (dcountprofile == null)
	              {
	                 htGoCountsProfile.put(szgoid, new Double(davgweight));
	              }
                      else
	              {
	      	         htGoCountsProfile.put(szgoid, 
                                  new Double(dcountprofile.doubleValue()+davgweight));
		      }
		   }
	        }
             }
	     //call procedure to update hIDpvalProfile with any better p-values
	     updateBestHyperBinom(htGoCountsProfile, hIDpvalProfile, 
				  numtotalgenes, htFullCount,dselectprofile,
                                  theDataSet.expectedassignments[nprofile]);          
          }
	  //call procedure to update hIDpval with any better p-values
	  updateBest(htGoCounts, hIDpval, numtotalgenes, htFullCount,dselect);
       }

       //go through any profiles missed
       for (int nprofile = 0; nprofile < banalyzed.length; nprofile++)
       {
          if (!banalyzed[nprofile])
	  {
	     double dselectprofile = theDataSet.countassignments[nprofile];
	     HashMap htGoCountsProfile = new HashMap();
             int npasize = theDataSet.profilesAssigned[nprofile].size();
             for (int nindex = 0; nindex < npasize; nindex++)
	     {
                int ngeneindex = ((Integer) theDataSet.profilesAssigned[nprofile].get(nindex)).intValue();
                String szGene =  genesprobes[ngeneindex];

                HashSet goList = labelsForID(szGene);
		Iterator goListitr = goList.iterator();
           
                double davgweight = 1/(double) theDataSet.bestassignments[ngeneindex].size();
		 
                while (goListitr.hasNext())
	        {
	           String szgoid = (String) goListitr.next();
	           Double dcountprofile = (Double) htGoCountsProfile.get(szgoid);
                   if (dcountprofile == null)
	           {
	              htGoCountsProfile.put(szgoid, new Double(davgweight));
	           }
                   else
	           {
	      	      htGoCountsProfile.put(szgoid, 
                                  new Double(dcountprofile.doubleValue()+davgweight));
		   }
	        }
             }        
	     //call procedure to update hIDpvalProfile with any better p-values 
	     updateBestHyperBinom(htGoCountsProfile, hIDpvalProfile, 
				  numtotalgenes, htFullCount,dselectprofile,
                                  theDataSet.expectedassignments[nprofile]);  
	  }
       }

       int nsize = hIDpval.keySet().size();
       int nsizeprofile = hIDpvalProfile.keySet().size();
       theRecIDpval = new RecIDpval2[nsizeprofile];
       theClusterRecIDpval = new RecIDpval[nsize];

       //converts the hashtables into arrays
       hashtoList1(hIDpval,theClusterRecIDpval);
       hashtoList2(hIDpvalProfile,theRecIDpval);
    }

}