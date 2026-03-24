package edu.cmu.cs.sb.chromviewer;

import edu.cmu.cs.sb.core.*;
import java.io.*;
import java.net.URL;
import java.net.URLConnection;
import java.net.URLEncoder;
import java.util.*;
import java.util.zip.*;

/**
 * Class responsible for downloading gene locations from Biomart
 */
public class BiomartAccess 
{
    /**
     *The URL of the biomart service
     */
    final static String SZBIOMARTURL = "http://www.biomart.org/biomart/martservice";

    /**
     *The maximum number of alias attributes BioMart allows for a single query
     */
    final static int NMAXATTR = 3;

    /**
     *This is a file handle on the biomart_species.txt file
     */
   private File m_file = null;

    /**
     *The number of lines we have read so far
     */
   private int nline;

    /**
     *The number of unique records that the count query returns 
     *times the number of times we execute a retrieval query
     *Note that the number of lines in the output maybe larger than this
     *but it is not feasible to get a count on the actual total number lines    
     */
   private int numrecs;

    /**
     *The number of unique records that the count query returns
     */
   private int nsingleset;

    /**
     *Constructor, sets m_file to a File handle on fname
     */
   public BiomartAccess(String fname) 
   {
      m_file = new File(fname);
   }

    /**
     *Executes the BioMart query in xmlMessage that counts the number of records
     *satisfying query. Returns this count of records.
     *This is used for the progress download.
     */
   public int executeCountQuery(String xmlMessage) throws IOException 
   {
      URL biomartServer = new URL(SZBIOMARTURL);
      URLConnection conn = biomartServer.openConnection();
	
      // use POST method
      conn.setDoOutput(true);
      OutputStreamWriter osw = new OutputStreamWriter(conn.getOutputStream());
      osw.write("query=" + URLEncoder.encode(xmlMessage, "UTF-8"));
      osw.flush();    
      
      // read the response into a file
      BufferedReader brresponse = new BufferedReader(new InputStreamReader(conn.getInputStream()));	
      String ln;

      ln = brresponse.readLine();
      brresponse.close();
      osw.close();
      int numlines;

      try
      {
         numlines = Integer.parseInt(ln);
      }
      catch (NumberFormatException nfex)
      {
	 throw new IllegalArgumentException("Count query did not return a number, instead returned "+ln);
      }

      if (numlines < 1)
      {
         throw new IllegalArgumentException("Count query returned "+numlines+" records. The number of "+
                                             "records must be positive!");
      }

      return numlines;
   }

    /**
     *ExecuteQuery. xmlMessage is the string containing the BioMart query to execute.
     *bos is the the output stream where the results of xmlMessage is written
     *Updates npercentdone[nstatusfield] after obtaining lock on lockpd
     */
   public void executeQuery(BufferedOutputStream bos, String xmlMessage,
        int[] npercentdone,int nstatusfield, Object lockpd) throws IOException 
   {

      URL biomartServer = new URL(SZBIOMARTURL);
      URLConnection conn = biomartServer.openConnection();
	
      // use POST method
      conn.setDoOutput(true);
      OutputStreamWriter osw = new OutputStreamWriter(conn.getOutputStream());
      osw.write("query=" + URLEncoder.encode(xmlMessage, "UTF-8"));
      osw.flush();    
	
      // read the response into a file
      BufferedReader brresponse = new BufferedReader(new InputStreamReader(conn.getInputStream()));
	
      String ln;
      int nlastpercentdone = 0;
      int nleft = nsingleset;
      String szeol = System.getProperty("line.separator");
      while ((ln = brresponse.readLine()) != null) 
      {
         String lnend = ln+szeol;
	 if (lnend.startsWith("Query ERROR"))
	 {
	     //not going to bail just because cannot get locations
	     System.out.println("Gene Location Download Error: "+lnend);
	 }
	 byte[] lnb = lnend.getBytes();
         bos.write(lnb,0,lnb.length);
     
         if (nleft > 0)
	 {
	     //the actual number of lines can be greater than the number of records
	     //we will not count these extra lines
	    nleft--;
	    nline++;
         }

	 synchronized (lockpd)
	 {
	    int nval = (int)  (100*nline/(double) numrecs);
	    if (nval <= 99)
	    {
		//we don't want to display 100, because we are not really done
	       npercentdone[nstatusfield] =nval;
	       if ((nlastpercentdone != npercentdone[nstatusfield]))
	       {
		   //only notify interface if percentage has changed
                  nlastpercentdone = npercentdone[nstatusfield];
                  lockpd.notifyAll();
      	       }
	    }
	 }  
      }
	
      brresponse.close();
      osw.close();
   }

   ///////////////////////////////////////////////////////////////////////////
   /**
    * Gets a HashSet of valid attributes for the gene location
    */
   public HashSet getValidAttributes(String szdataset) throws IOException
   {
      HashSet hs = new HashSet();
      String xmlMessage;

      String szMetaQuery ="?virtualSchema=default&type=attributes&dataset="+szdataset;
      URL biomartServer = new URL(SZBIOMARTURL+szMetaQuery);
      URLConnection conn = biomartServer.openConnection();
	
      // use POST method 
      BufferedReader brresponse = new BufferedReader(new InputStreamReader(conn.getInputStream()));
      String szLine;
      while ((szLine = brresponse.readLine())!=null)
      {
         StringTokenizer st = new StringTokenizer(szLine,"\t");
         if (st.hasMoreTokens())
	 {
            String szattribute = st.nextToken();
            hs.add(szattribute);
         }
      }
      brresponse.close();
      return hs;
   }

   /**
   *Species is the name of the species to download the locations for.
   *Updates npercentdone[nstatusfield] with status, first gets lock on lockpd
   */
   public void updateGeneFile(String species,String szoutfile,int[] npercentdone,int nstatusfield,
			     Object lockpd) throws IOException 
   {	 
      // try opening the configuration file
      BufferedReader brspecies = new BufferedReader(new FileReader(m_file));
	    
      // read line by line and try to find the species
      String ln = "";
      String[] fields = null;
      boolean bfound = false;
      while ((!bfound)&&((ln = brspecies.readLine()) != null)) 
      {
         fields = ln.split(",");
	 String[] szIDpre = fields[0].split("_");
	 // is this the right species?
	 if (szIDpre[0].equalsIgnoreCase(species))
	 {
	    bfound = true;
	 }
      }
      // close the file
      brspecies.close();

      if (!bfound) 
      {
	  //species was not found throw an errir
         throw new IllegalArgumentException("The species provided - " + species + " - is not found in "+m_file);
      } 

      if (fields.length < 2)
      {
	  throw new IllegalArgumentException("Line "+ln+" in "+m_file+" only has one entry, no delimiting ',' found");
      } 
	       
      String xmlMessagePre, //The initial part of the message needed for both 
	     xmlMessagePost, //The part of the message immediately after the query params
	                     //These are the non-alias attributes
 	     xmlMessageEnding, //The last part of the message needed
	     xmlMessageCount1, //Specifies query params where count=1 
	                       //meaning just a count of numrecords satisfying query is returned
	     xmlMessageCountEmpty; //Specifies query params where count is empty meaning 
                                   //the query actually returns the records

      // construct the xml message for the request
      xmlMessagePre = "<?xml version = \"1.0\" encoding = \"UTF-8\"?>";
      xmlMessagePre += "<!DOCTYPE Query>";

      xmlMessageCount1 = "<Query virtualSchemaName = \"default\" uniqueRows = \"1\" count = \"1\" softwareVersion = \"0.6\" >";
      xmlMessageCountEmpty = "<Query virtualSchemaName = \"default\" uniqueRows = \"1\" count = \"\" softwareVersion = \"0.6\" >";

      xmlMessagePost = "<Dataset name = \"" + fields[0] + "\" interface = \"default\" >";
      xmlMessagePost += "<Attribute name = \"chromosome_name\" />";
      xmlMessagePost += "<Attribute name = \"start_position\" />";
      xmlMessagePost += "<Attribute name = \"end_position\" />";
      xmlMessagePost += "<Attribute name = \"strand\" />";
      xmlMessagePost += "<Attribute name = \"ensembl_gene_id\" />";

      xmlMessageEnding = "</Dataset>";
      xmlMessageEnding += "</Query>";

      String xmlMessageCount = xmlMessagePre + xmlMessageCount1 + xmlMessagePost + xmlMessageEnding;
      nsingleset = executeCountQuery(xmlMessageCount);

      // output file writer
      BufferedOutputStream bos = new BufferedOutputStream(new
	    			   GZIPOutputStream(new FileOutputStream(szoutfile)));
      //write in the species name in the first line of the file
     
      //writes the species field to the file 
      String szeol = System.getProperty("line.separator");
      String szspecieseol = fields[1] + szeol;
      byte[] lnb = szspecieseol.getBytes();
      bos.write(lnb,0,lnb.length);


      String xmlMessage =  xmlMessagePre + xmlMessageCountEmpty + xmlMessagePost;
      // messages can contain at most 3 alias attributes
      // if we have more than that, need too break
      // up into several requests
      boolean run_last = true;
      nline = 0; //we haven't read any lines yet
      String query;

      HashSet hsvalidAttributes = getValidAttributes(fields[0]);

      // add in species-specific gene ids
      if (fields.length > 2) 
      {
         String[] aliases = fields[2].split(":");
	 numrecs = nsingleset*((int) Math.ceil(aliases.length/(double)NMAXATTR));
	 query = xmlMessage;
	 for (int i = 0; i < aliases.length; i++) 
         {
	    if (hsvalidAttributes.contains(aliases[i]))
	    { 
	       query += "<Attribute name = \"" + aliases[i] + "\" />";
	       // a query is being constructed.  It might
	       // need to be run through the last request
	       run_last = true;
		 
	       if (i % NMAXATTR == (NMAXATTR-1)) 
               {
		  //last alias attribute of the query, going to execute it
	          query += xmlMessageEnding;
	          // close the xml doc
	          executeQuery(bos, query,npercentdone,nstatusfield,lockpd);
	
	          // reset query for reuse
	          query = xmlMessage;
	          // all attributes so far have been processed
	          // no need to run the last request
	          run_last = false;
	       }
	    }
	 }
      }
      else
      {
         query =  xmlMessage;
      } 
	    
      // last request.  Picks up everything not processed by loop above.
      // including if the loop above wasn't executed at all because
      // there were no extra aliases to query.
      if (run_last)
      {
	 query += xmlMessageEnding;	
	 executeQuery(bos, query,npercentdone,nstatusfield,lockpd);
      }
      // close the xml doc	
      bos.close();	
   }
}	    
	    
	    
	    
	    
	   
