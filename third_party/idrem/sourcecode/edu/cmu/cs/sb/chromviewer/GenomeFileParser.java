package edu.cmu.cs.sb.chromviewer;

import edu.cmu.cs.sb.core.*;
import java.io.*;
import java.util.*;
import java.util.zip.*;

import java.awt.Color;

/**
 * Class responsible for parsing and storing a file of the location of chromosome location of genes
 */
public class GenomeFileParser 
{    

   public String szchromval;
   public String szchromsourceval;

    /**
     * Association of all gene names with gene structures
     */
   private Hashtable m_allGenes = null;

    /**
     * Genes considered valid
     */
    private HashSet hs_baseGenes = null;

    /**
     * Association of possible gene aliases with gene structures
     */
   private Hashtable m_aliasedGenes = null;


   private HashSet hs_activeGenes = null;

   Gene[] m_chromArray = null;

   private boolean m_parsedOK = false;

   private String m_species = "";

   private int nchromLenMax;
    
   int[] sortMappingID;
   int[] sortMappingSize;
   HashSet[] basegenesets;    

   boolean brandcorrectedpval;
   int nsamplespval;

    /**
     * Record for a chromosome ID
     */
   static class ChromIDRec
   {
      int ntype; //0 roman, 1 numeric, 2 string
      int nchrom;
      String szchrom;
      int norigindex;
   }

   static int TYPE_ROMAN = 0;
   static int TYPE_NUMERIC = 1;
   static int TYPE_STRING = 2;

    /**
     * Empty constructor
     */
   public GenomeFileParser()
   {
   }
   

    /**
     * Constructor that attempts to load
     */
   public GenomeFileParser(
	   String szchromval, String szchromsourceval,
           String[] baseGenes,
	   ArrayList alExtraGenes,boolean brandcorrectedpval,int nsamplespval)
                                    throws IOException
   {
      this.szchromval = szchromval.trim();
      this.szchromsourceval = szchromsourceval;

      if (!szchromval.equals(""))
      {
         m_allGenes = new Hashtable();
         m_aliasedGenes = new Hashtable();

         this.brandcorrectedpval = brandcorrectedpval;
         this.nsamplespval = nsamplespval;

         File f = new File(szchromval);
    
         String fn = f.getName();

         String[] sa = fn.split("\\.");
         m_species = sa[0];
         String im = sa[1];

         BufferedReader brgenefile;	    
         try
         {
            //first tries gzip format, if that fails then does regular 
            brgenefile = new BufferedReader(new InputStreamReader(
                             new GZIPInputStream(new FileInputStream(f))));
         }
         catch (IOException ex)
         {
            brgenefile = new BufferedReader(new FileReader(f));
         }
	    
         Hashtable chromHash = new Hashtable();
  
         if (im.startsWith("gff"))
         {
	    m_parsedOK = parseGff(brgenefile,chromHash);
         }
         else if (im.startsWith("mart"))
         {
	    m_parsedOK = parseBioMart(brgenefile,chromHash);
         }
         else
         {
            m_parsedOK = false;
	    throw new IllegalArgumentException(fn+"is an invalid location file name. Beginning or entire file name "+
                      "should have the form 'species.gff' or 'species.mart'");
         }

         brgenefile.close();
         // if the file was empty, consider parsing failed
         if (chromHash.size() == 0)
         {
            throw new IllegalArgumentException("No gene entries found in file "+szchromval);
         }

         sortMappingID =sortChromsID(chromHash);
         sortMappingSize =sortChromsSize(chromHash);
         // convert chromosomes from a hash table keyed on name to an array
         m_chromArray = new Gene[chromHash.size()];
 
         // chromosome number is updated to the sorted position of that
         for (Enumeration e = chromHash.elements(); e.hasMoreElements(); ) 
         {
            Gene g =(Gene)  e.nextElement();
	    m_chromArray[g.chromosome] = g;
         }
	    
         hs_activeGenes = new HashSet();
         hs_baseGenes = new HashSet();
         HashSet[] tempgenesets = addBaseGenes(baseGenes,alExtraGenes);
         int ncount = 0;
         for (int nindex = 0; nindex < tempgenesets.length; nindex++)
         {
            if (tempgenesets[nindex]!= null)
	    {
               ncount++;
            }
         }

         basegenesets = new HashSet[ncount];
         int nactualindex = 0;
         for (int nindex = 0; ((nactualindex < basegenesets.length)&&
	                    (nindex < tempgenesets.length)); nindex++)
         {
	    if (tempgenesets[nindex]!= null)
	    {
	       basegenesets[nactualindex] = tempgenesets[nindex];
	       nactualindex++;
	    }
	 } 
      }
   }

    /**
     * Compares two chromosome IDs first by type, then ID
     */
   static class ChromIDRecCompare implements Comparator
   {
      public int compare(Object o1, Object o2)
      {
         ChromIDRec cIDrec1 = (ChromIDRec) o1;
	 ChromIDRec cIDrec2 = (ChromIDRec) o2;

         if (cIDrec1.ntype < cIDrec2.ntype)
         {
            return -1;
	 }
	 else if (cIDrec1.ntype > cIDrec2.ntype)
	 {
	    return 1;
	 }
	 else if (cIDrec1.ntype == TYPE_STRING)
	 {
	    return cIDrec1.szchrom.compareTo(cIDrec2.szchrom);
	 }
	 else
	 {
	    if (cIDrec1.nchrom < cIDrec2.nchrom)
	    {
	       return -1;
	    }
	    else if (cIDrec1.nchrom > cIDrec2.nchrom)
	    {
	       return 1;
	    }
	    else
	    {
               return 0;
	    }
	 }
      }
   }

    /**
     * Chromsome record containing its length and index
     */
    static class ChromLengthRec
    {
	int nlength;
	int norigindex;
    }

    /**
     * Sorts chromosomes by length and then index
     */
    static class ChromLengthRecCompare implements Comparator
    {
       public int compare(Object o1, Object o2)
       {
          ChromLengthRec clr1 = (ChromLengthRec) o1;
          ChromLengthRec clr2 = (ChromLengthRec) o2;

          if (clr1.nlength > clr2.nlength)
          {
      	     return -1;
	  }
	  else if (clr1.nlength < clr2.nlength)
          {
	     return 1;
	  }
	  else if (clr1.norigindex < clr2.norigindex)
          {
	     return -1;
	  }
	  else if (clr1.norigindex > clr2.norigindex)
          {
	     return 1;
	  }
	  else
          {
	     return 0;
	  }
       }
    }


    /**
     *Given a chromHash maps each chromosome to a new ID based on the
     *ordering numeric, then roman, then strings increasing within these lists
     */
   private int[] sortChromsID(Hashtable chromHash) 
   {
       ChromIDRec[] cid = new ChromIDRec[chromHash.size()];
       Enumeration els = chromHash.elements();
       for (int nindex = 0; nindex < cid.length; nindex++)
       {
          Gene theGene =(Gene) els.nextElement();
	  cid[nindex] = new ChromIDRec();
          cid[nindex].norigindex = theGene.chromosome;
          cid[nindex].szchrom = theGene.namechrom;

	  if (theGene.namechrom.equals("X"))
	  {
	      cid[nindex].ntype = TYPE_STRING;
	      cid[nindex].szchrom = theGene.namechrom;
	  }
	  else if (theGene.namechrom.matches("^\\d+$")) 
          {
	      cid[nindex].ntype = TYPE_NUMERIC;
	      cid[nindex].nchrom = Integer.parseInt(theGene.namechrom);
	  }
	  else if (theGene.namechrom.matches("^[iIvVxXlL]+$")) 
          {
	      cid[nindex].ntype = TYPE_ROMAN;
	      cid[nindex].nchrom = Util.romanToNumeric(theGene.namechrom);
	  }
	  else
	  {
	      cid[nindex].ntype = TYPE_STRING;
	      cid[nindex].szchrom = theGene.namechrom;
	  }

       }
       Arrays.sort(cid, new ChromIDRecCompare());
      
       int[] sorted = new int[cid.length];
       for (int nindex = 0; nindex < cid.length; nindex++)
       {
	   sorted[cid[nindex].norigindex] = nindex;
       }
       
       return sorted;
   }

    /**
     * Sorts chromosome by length using ChromLengthRecCompare
     */ 
   private int[] sortChromsSize(Hashtable chromHash) 
   {
       ChromLengthRec[] clr = new ChromLengthRec[chromHash.size()];
       Enumeration els = chromHash.elements();
       for (int nindex = 0; nindex < clr.length; nindex++)
       {
	   Gene theGene =(Gene) els.nextElement();
	   clr[nindex] = new ChromLengthRec();
	   clr[nindex].nlength = theGene.end;
	   clr[nindex].norigindex = theGene.chromosome;
       }
       Arrays.sort(clr, new ChromLengthRecCompare());
      
       int[] sorted = new int[clr.length];
       for (int nindex = 0; nindex < clr.length; nindex++)
       {
	   sorted[clr[nindex].norigindex] = nindex;
       }
       
       return sorted;
   }


    /**
     * Parses a biomart file
     */
   private boolean parseBioMart(BufferedReader br,Hashtable chromHash) throws IOException 
   {	
      String fieldSeparator = "\t";	
	
      int CHROM_NAME  = 0;
      int START_COORD = 1;
      int END_COORD   = 2;
      int STRAND      = 3;
      int GENE_NAME   = 4;

      // the biomart files will not have any special chromosome entries
      // all information must be inferred from the gene data
      // length of each chromosome will be determined by the location of the last gene

      nchromLenMax = 0;
      //Hashtable chromHash = new Hashtable();

      // first line of the file contains the species name
      m_species = br.readLine();
	
      String ln;
      String[] fields;
      // now go through the main loop and read in the genes
      while ((ln = br.readLine()) != null) 
      {
         fields = ln.split(fieldSeparator);
	    
         // is this chromosome already known?
	 Gene chr =(Gene)  chromHash.get(fields[CHROM_NAME]);
	 if (chr == null) 
         {
	    // if not make a new one
	    // Gene is really a chromomsome here
	    chr = new Gene();

	    chr.namechrom = fields[CHROM_NAME];
	    chr.start = 1;
	    chr.end = 1;
	    chr.chromosome = chromHash.size();
	    chromHash.put(fields[CHROM_NAME], chr);
	 }     
	    
	 char chstrand = ((Integer.parseInt(fields[STRAND]) == -1) ? '-' : '+');
	 String szloc = chr.chromosome+"|"+fields[START_COORD]+"|"+fields[END_COORD]+"|"+chstrand;
	 // now that we have the chromosome look at the gene
	 // have we already seen it?
	 Gene gene = (Gene) m_allGenes.get(szloc);
	 if (gene == null) 
         {
	    gene = new Gene();
	    gene.chromosome = chr.chromosome;
	    int ncurrend = Integer.parseInt(fields[END_COORD]);
	    gene.start = Integer.parseInt(fields[START_COORD]);
	    gene.end = ncurrend;

            // if this is the gene closest to end of chromosome so far, update chrom length
            if (ncurrend > chr.end)
	    {
	       chr.end = ncurrend;
	       if (ncurrend > nchromLenMax)
	       {
		  nchromLenMax =ncurrend;
	       }
	    }

	    gene.strand =chstrand;
	    // insert this new gene into the list
	    m_allGenes.put(szloc, gene);
	 }

	 //map gene symbol to arraylist of locations
	 HashSet hsGenes = (HashSet) m_aliasedGenes.get(fields[GENE_NAME]);	
	 if (hsGenes == null)
	 {
            hsGenes = new HashSet(1);
	    m_aliasedGenes.put(fields[GENE_NAME],hsGenes);
	 }

	 hsGenes.add(gene);
	 // put all aliases including official name of this gene into the alias list
	 //m_aliasedGenes.put(gene.name, gene);
	 for (int i = GENE_NAME; i < fields.length; i++) 
         {
	    // some aliases might not have a value
	    if (fields[i].length() == 0)
	       continue;

	    // we might have already seen this alias
	    hsGenes = (HashSet) m_aliasedGenes.get(fields[i]);
	    if (hsGenes == null)
	    {
		hsGenes = new HashSet(1);
		m_aliasedGenes.put(fields[i],hsGenes);
	    }
	    hsGenes.add(gene);		
	 }	    
      }

      // if the file was empty, consider parsing failed
      if (chromHash.size() == 0)
      {
         throw new IllegalArgumentException("No gene entries found in the biomart location file");
      }

      return true;
   }
 
    /**
     * Parses a GFF format file
     */
    private boolean parseGff(BufferedReader br,Hashtable chromHash) throws IOException 
   {
      // file format: awk '{ if($3 == "gene" || $3 == "chromosome") print $0 }' saccharomyces_cerevisiae.gff | awk -F';' '{ print $1 }' | sed 's/ID=//' 
      // CHROM_NAME ORGANISM LINE_TYPE START_COORDINATE END_COORDINATE . STRAND . NAME
      final int CHROM_NAME  = 0;
      final int LINE_TYPE   = 2;
      final int START_COORD = 3;
      final int END_COORD   = 4;
      final int STRAND      = 6;
      final int GENE_NAME   = 8;


      String ln;
      while ((ln = br.readLine()) != null) 
      {
	  
         if (ln.startsWith("##FASTA"))
	 {
	     //indicates rest is just sequence
            break;
	 }

         if (ln.startsWith("#"))
	 {
            // lines starting with # are comments
	    continue;
	 }

         String[] fields = ln.split("\\t"); 
         //tab since white space allowed in gene names
	    
         if (fields[LINE_TYPE].equals("chromosome")) 
         {
	    String szchromname = fields[CHROM_NAME];

	    Gene chr =(Gene)  chromHash.get(szchromname);
	    if (chr == null) 
            {
	       // if not make a new one
	       // Gene is really a chromomsome here
	       chr = new Gene();
	       chr.namechrom = szchromname;
	       chr.start = Integer.parseInt(fields[START_COORD]);
	       chr.end = Integer.parseInt(fields[END_COORD]);
	       if (chr.end > nchromLenMax)
	       {
	          nchromLenMax = chr.end;
	       }
	       chr.chromosome = chromHash.size();
	       chromHash.put(chr.namechrom,chr);
	    }
	    else
	    {
	       int ncurrend = Integer.parseInt(fields[END_COORD]);
               if (ncurrend > chr.end)
	       {
	          chr.end = ncurrend;
	          if (ncurrend > nchromLenMax)
	          {
		     nchromLenMax =ncurrend;
	          }
	       }	        	       
	    }
	    
         }
         else if (fields[LINE_TYPE].equals("gene")) 
         {
	    Gene chr =(Gene)  chromHash.get(fields[CHROM_NAME]);
	    if (chr == null) 
            {
	       // if not make a new one
	       // Gene is really a chromomsome here
	       chr = new Gene();

	       chr.namechrom = fields[CHROM_NAME];
	       chr.start = 1;
	       chr.end = 1;
	       chr.chromosome = chromHash.size();
	       chromHash.put(fields[CHROM_NAME], chr);
	    }    
	    char chstrand = fields[STRAND].charAt(0);
 	    String szloc = chr.chromosome+"|"+fields[START_COORD]+"|"+fields[END_COORD]+"|"+chstrand;
	    // now that we have the chromosome look at the gene
	    // have we already seen it?
	    Gene gene = (Gene) m_allGenes.get(szloc);
	    if (gene == null) 
            {
	       gene = new Gene();
	       gene.chromosome = chr.chromosome;
	       int ncurrend = Integer.parseInt(fields[END_COORD]);
	       gene.start = Integer.parseInt(fields[START_COORD]);
	       gene.end = ncurrend;

               // if this is the gene closest to end of chromosome so far, update chrom length
               if (ncurrend > chr.end)
	       {
	          chr.end = ncurrend;
	          if (ncurrend > nchromLenMax)
	          {
		     nchromLenMax =ncurrend;
	          }
	       }

	       gene.strand =chstrand;
	       // insert this new gene into the list
	       m_allGenes.put(szloc, gene);
	    }
		
            // ID=XXXX;Name=XXXX;Alias=XXXX,XXXX,XXXX...
	    String[] infoSplit = fields[GENE_NAME].split(";");
	    for (int i = 0; i < infoSplit.length; i++) 
            {
               String[] dataSplit = infoSplit[i].split("=");
               if ((dataSplit[0].compareTo("ID") == 0)||(dataSplit[0].compareTo("Name")==0))
               {
	          //map gene symbol to arraylist of locations
	          HashSet hsGenes = (HashSet) m_aliasedGenes.get(dataSplit[1]);	
	          if (hsGenes == null)
	          {
                     hsGenes = new HashSet(1);
		     m_aliasedGenes.put(dataSplit[1],hsGenes);
		  }
	          hsGenes.add(gene);
	       }
	       else if (dataSplit[0].compareTo("Alias") == 0) 
               {
	          String[] listSplit = dataSplit[1].split(",");
	          for (int j = 0; j < listSplit.length; j++)
	          {
	             //map gene symbol to arraylist of locations
	             HashSet hsGenes = (HashSet) m_aliasedGenes.get(listSplit[j]);	
	             if (hsGenes == null)
	             {
                        hsGenes = new HashSet(1);
		        m_aliasedGenes.put(listSplit[j],hsGenes);
		     }
	             hsGenes.add(gene);
	          }
	       }
	    } // for...		
	 } // else if
      } // while
      return true;
   }
   
    /**
     * Variable accessor method
     */
   public String getSpecies() 
   {
       return m_species;
   }

    /**
     * Variable accessor method
     */
   public Gene[] getChromArray() 
   {
       return m_chromArray;
   }

    /**
     * Variable accessor method
     */
   public int getChromLenMax()
   {
	return nchromLenMax;
   }

    /**
     * Variable accessor method
     */
   public Gene[] getChroms() 
   {
      if (!m_parsedOK)
         return null;

      return m_chromArray;
   }
    
    /**
     * Variable accessor method
     */
   public boolean parsedOK() 
   {
      return m_parsedOK;
   }


    /**
     * Clears empty genes
     */
   public void clearActiveGenes() 
   {
       hs_activeGenes = new HashSet();
   }


    /**
     * Adds to the base set of genes all genes in input names and alnames
     */
   public HashSet[] addBaseGenes(String[] inputnames,ArrayList alnames)
   {
       if (alnames == null)
       {
           return addGenesHelper(inputnames,hs_baseGenes);
       }
       else
       {
          int nsize = alnames.size();
          String[] names = new String[inputnames.length+nsize];
          for (int nindex = 0; nindex < inputnames.length; nindex++)
          {
             names[nindex] =inputnames[nindex];
          }

	  for (int nindex = 0; nindex < nsize; nindex++)
          {
             names[nindex+inputnames.length] = (String) alnames.get(nindex);
          }
          return addGenesHelper(names,hs_baseGenes);
	}
    }

    /**
     * Adds the set of genes in names to those currently displayed by the chromosome viewer
     */
    public HashSet[] addActiveGenes(String[] names)
    {
	return addGenesHelper(names,hs_activeGenes);
    }

   /**
    *   specify the set of active genes in the experiment
    */
   private HashSet[] addGenesHelper(String[] names,HashSet hsinput) 
   {
      HashSet g = null;

      HashSet[] out = new HashSet[names.length];
	
      for (int i = 0; i < names.length; i++) 
      {
         out[i] = null;

         // the name might come in as a list of aliases
         // separated by some delimiter.
         // try each alias.
	 g = (HashSet) m_aliasedGenes.get(names[i]);
	 if (g != null)
	 {
	    out[i] =g;
	    hsinput.addAll(out[i]);
	 }
	 else
	 {

            String nameList[] = names[i].split("[;|,]");
	    
            for (int j = 0; j < nameList.length; j++) 
            {
	       // first get the unaliased name
	       g = (HashSet) m_aliasedGenes.get(nameList[j]);
	     
               // query on that name
               if (g != null) 
	       {	    
                  out[i] = g; //take first match
	          hsinput.addAll(out[i]);
                  break;
	       }
	    }
         }	
      }

      return out;
   }


        
   /**
    *
    *   go through all genes in the base gene hashtable
    *   and note the distribution over the chromosomes
    */
   int[] getBaseCountPerChrom() 
   {	
       return getCountPerChromHelper(hs_baseGenes);
   }

    /**
     *   go through all genes in the base gene hashtable
     *   and note the distribution over the chromosomes
     */
   int[] getActiveCountPerChrom() 
   {
       return getCountPerChromHelper(hs_activeGenes);
   }

    /**
     * Tallys the number of genes on each chromosome in hsinput
     */
    private int[] getCountPerChromHelper(HashSet hsinput)
    {
      if (!m_parsedOK)
         return null;

      int[] chromCounts = new int[m_chromArray.length];
      // initialize array
      for (int i = 0; i < chromCounts.length; i++)
      {
         chromCounts[i] = 0;
      }

      Iterator hsitr = (Iterator) hsinput.iterator();
      while (hsitr.hasNext())
      {
         Gene currGene = (Gene) hsitr.next();
	 chromCounts[currGene.chromosome]++;	 
      }


      return chromCounts;
   }

}
