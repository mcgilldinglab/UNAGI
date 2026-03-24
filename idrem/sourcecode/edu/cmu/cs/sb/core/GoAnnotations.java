package edu.cmu.cs.sb.core;

import java.util.*;
import java.io.*;
import java.util.zip.*;
import java.text.*;

/**
 *The class encapsulates the data and functions related to the Gene Ontology
 * and gene set enrichment analysis
 * 
 * @author Jason Ernst
 */
public class GoAnnotations {

	public static String SZGP = "gene_association.";

	//only user prvided cross reference files are supported
	public static String[] defaultxrefs = { "User provided",
			"No cross references"};

	public static String[] organisms = {
			"User provided",
			"No annotations",
			//"Anaplasma phagocytophilum HZ (JCVI)",
			//"Agrobacterium tumefaciensstr. C58 (PAMGO)",// new atumefaciens
			"Arabidopsis (EBI)",
			"Arabidopsis thaliana (TAIR)",
			"Aspergillus nidulans (AspGD)",// new aspgd.gz
			//"Bacillus anthracis (JCVI)",
			"Caenorhabditis elegans (WormBase)",
			//"Campylobacter jejuni RM1221 (JCVI)",
			"Candida albicans (CGD)",
			//"Carboxydothermus hydrogenoformans Z-2901 (JCVI)",
			"Chicken (EBI)",
			"Comprehensive Microbial Resource (JCVI)",
			//"Clostridium perfringens ATCC13124 (JCVI)",// new cperfringens
			//"Colwellia psychrerythraea 34H (JCVI)",// new cpsychreythrea
			"Cow (EBI)",
			//"Coxiella burnetti RSA (JCVI)",
			"Danio rerio (ZFIN)",
			//"Dehalococcoides ethenogenes 195 (JCVI)",
			"Dictyostelim discoideum (DictyBase)",// now dictyBase.gz
			"Dog (EBI)",//new DOG EBI
			"Dickeya dadantii (PAMGO)", // added version 1.3.7

			"Drosophila melanogaster (FlyBase)",
			//"Ehrlichia chaffeensis Arkansas (JCVI)",
			"Escherichia coli (PortEco)",// new ecocyc.gz
			//"Geobacter sulfurreducens PCA (JCVI)",
			// "Glossina morsitans (Sanger GeneDB)", removed version 1.37
			"Fly (EBI)", //new
			"Human (EBI)",
			//"Hyphomonas neptunium ATCC 15444 (JCVI)",// new hneptunium
			"Leishmania major (Sanger GeneDB)",
			//"Listeria monocytogenes 4b F2365 (JCVI)",
			"Magnaporthe grisea (PAMGO)",// new mgrisea
			//"Methylococcus capsulatus Bath (JCVI)",
			"Mouse (EBI)",
			"Mus musculus (MGI)",
			//"Neorickettsia sennetsu Miyayama (JCVI)",
			"Oomycetes (PAMGO)", // new oomycetes
			"Oryza sativa (Gramene)",
			"PDB (EBI)",
			"Pig (EBI)", //new
			"Plasmodium falciparum (Sanger GeneDB)",
			"Pseudomonas aeruginosa PA01 (PseudoCAP)",
			//"Pseudomonas fluorescens Pf-5 (JCVI)",// new tigr_Pfluorescens.gz
			//"Pseudomonas syringae DC3000 (JCVI)",
			//"Pseudomonas syringae pv. phaseolicola 1448A (JCVI)",// new phaseolicola
			"Rat (EBI)",
			"Rattus norvegicus (RGD)",
			"Reactome (CSHL&EBI)",// new reactome
			"Saccharomyces cerevisiae (SGD)",
			"Schizosaccharomyces pombe (Sanger GeneDB)",
			//"Shewanella oneidensis MR-1 (JCVI)",
			//"Silicibacter pomeroyi DSS-3 (JCVI)", 
                        "Solanaceae (SGN)", // new sgn.gz
			"Trypanosoma brucei (Sanger GeneDB)",
			// "Trypanosoma brucei chr 2 (JCVI)", removed no longer available
			//"UniProt (EBI)",
			"Worm (EBI)", // new
			"Yeast (EBI)", // new
			// "UniProt no IEA (EBI)", //new uniprot_noiea
			//"Vibrio cholera (JCVI)", 
                        "Zebrafish (EBI)" };

	//no xref files are supported anymore because hardcoded links with EBI dont exist anymore
	public static String[] xreffile = { "", "", "",
			"", "",
			// "human.xrefs.gz", //Changed 7/5/09 human xrefs no longer
			// supported
			"", "", "" };

       public static String[] xreforgfile = { //"",
	   // "", 
	   //"", 
	   //              "",
			"", "", "", "", "", "", "", "",
			"", "", "", "", "", "", "", "", "", "",
			"", "", "", "",
			"",// removed 7/5/09 human xrefs no longer
				// supported"human.xrefs.gz",
			"", "", "", "", "", "", "", "", "", "", "", "", "",
			"", "", 
			//"", "", "", "", "", "", "", "", "", "",
			// "",
			//"", "" 
                         };

	public static String[] gannfile = {
			"",
			"",
			//SZGP + "jcvi_Aphagocytophilum.gz",
			//SZGP + "PAMGO_Atumefaciens.gz",// new
			"goa_arabidopsis.gaf.gz",
			SZGP + "tair.gz",
			SZGP + "aspgd.gz",// new
			//SZGP + "jcvi_Banthracis.gz",
			SZGP + "wb.gz",
			//SZGP + "jcvi_Cjejuni.gz",
			SZGP + "cgd.gz",
			//SZGP + "jcvi_Chydrogenoformans.gz",
			"goa_chicken.gaf.gz",
			SZGP+"jcvi.gz",
			//SZGP + "jcvi_Cperfringens.gz", // new
			//SZGP + "jcvi_Cpsychrerythraea.gz", // new
			"goa_cow.gaf.gz",
			//SZGP + "jcvi_Cburnetii.gz",
			SZGP + "zfin.gz",
			//SZGP + "jcvi_Dethenogenes.gz",
			SZGP + "dictyBase.gz",// new SZGP+"ddb.gz",
			"goa_dog.gaf.gz",
			SZGP + "PAMGO_Ddadantii.gz",
			SZGP + "fb.gz",
			//SZGP + "jcvi_Echaffeensis.gz",
			SZGP + "ecocyc.gz", // new
			"goa_fly.gaf.gz",//new
			//SZGP + "jcvi_Gsulfurreducens.gz",
			// SZGP+"GeneDB_tsetse.gz",
			"goa_human.gaf.gz",
			//SZGP + "jcvi_Hneptunium.gz",// new
			SZGP + "GeneDB_Lmajor.gz",
			//SZGP + "jcvi_Lmonocytogenes.gz",
			SZGP + "PAMGO_Mgrisea.gz",// new
			//SZGP + "jcvi_Mcapsulatus.gz",
			"goa_mouse.gaf.gz",
			SZGP + "mgi.gz",
			//SZGP + "jcvi_Nsennetsu.gz",
			SZGP + "PAMGO_Oomycetes.gz",// new
			SZGP + "gramene_oryza.gz",
			"goa_pdb.gaf.gz",
			"goa_pig.gaf.gz",
			SZGP + "GeneDB_Pfalciparum.gz",
			SZGP + "pseudocap.gz",
			//SZGP + "jcvi_Pfluorescens.gz",// new
			//SZGP + "jcvi_Psyringae.gz",
			//SZGP + "jcvi_Psyringae_phaseolicola.gz",// new
		        "goa_rat.gaf.gz",
			SZGP + "rgd.gz",
			SZGP + "reactome.gz",// new
			SZGP + "sgd.gz", 
                        SZGP + "pombase.gz",//updated from "GeneDB_Spombe.gz",
			//SZGP + "jcvi_Soneidensis.gz", SZGP + "jcvi_Spomeroyi.gz",
			SZGP + "sgn.gz",// new
			SZGP + "GeneDB_Tbrucei.gz",
			// SZGP+"tigr_Tbrucei_chr2.gz",
			//"goa_uniprot_all.gaf.gz",
			"goa_worm.gaf.gz",
			"goa_yeast.gaf.gz",
			// SZGP+"goa_uniprot_noiea.gz",
			//SZGP + "jcvi_Vcholerae.gz", 
                        "goa_zebrafish.gaf.gz" };

	/**
	 *Source of the GO annotations
	 */
	public String szorganismsourceval;

	/**
	 *Source of the cross-reference
	 */
	public String szxrefsourceval;

	/**
	 *Rec is a Gene Ontology hierarchy record a GO ID will map to the record A
	 * record contains a name and a set of parent records
	 */
	public static class Rec {
		/**
		 *A set of parents in the hierarchy
		 */
		public HashSet parents;

		/**
		 *The name of the category
		 */
		public String sztermName;

		/**
		 *Initializes parents and sztermName
		 */
		public Rec(HashSet parents, String sztermName) {
			this.parents = parents;
			this.sztermName = sztermName;
		}
	}

	/**
	 *The name of the file which maps category IDs to category names that are
	 * not official gene ontology categories.
	 */
	public String szcategoryIDval;

	/**
	 *The name of the file with additional genes to add to the base set.
	 */
	public String szextraval;

	/**
	 *The name of the file with cross-references equating two or more gene
	 * symbols.
	 */
	public String szxrefval;

	/**
	 *Maps a gene ID to a list of go categories for which it is annotated.
	 */
	public HashMap htGoLabels;

	/**
	 *Maps a category to the total number of genes annotated as belonging to
	 * the category.
	 */
	public HashMap htFullCount;

	/**
	 *Maps every gene name to a boolean as to whether or not it is selected in
	 * the define gene set menu.
	 */
	public HashMap htGeneNames;

	/**
	 *Maps go category ids to their records contains their name and parents
	 */
	public HashMap htGO;

	/**
	 *Maps alternative IDs to the ID used in the hierarchy
	 */
	public HashMap htAlt;

	/**
	 *Maps a GO ID to it set of ancestor IDs
	 */
	public HashMap htAncestors;

	/**
	 *Maps ids in the categoryID file to category names
	 */
	public HashMap htIDCategory;

	/**
	 *Whether the current order by ID are for a cluster of profiles (true) or
	 * profiles (false)
	 */
	public boolean bcluster = false;

	/**
	 *Contains the total number of GO categories
	 */
	public int numcategory = 0;

	/**
	 *The total number of non-duplicate genes on the array
	 */
	public int numtotalgenes = 0;

	/**
	 *Contains the maximum number of genes in any category
	 */
	public int nmaxsize;

	/**
	 *Stores a count of the number
	 */
	public int nGeneSet = 0;

	/**
	 *The number of GO Categories
	 */
	public int nlegalgo = 0;

	/**
	 *A string containing the current sort command
	 */
	public String szsortcommand = "default";

	/**
	 *A string containing the GO ID the profiles are currently sorted by
	 */
	public String szSelectedGO = null;

	/**
	 *boolean as to whether the profile enrichment should be actual size based
	 * (true) or expected size based (false)
	 */
	public boolean bactual = true;

	/**
	 *The number a samples to use when the multiple hypothesis correction is
	 * based on randomization.
	 */
	public int nsamplespval = 500;

	/**
	 *The minimum number of genes in a set and belonging to a category for the
	 * category to be included in a GO analysis table
	 */
	public int nmingo;

	/**
	 *The minimum GO level to include inthe analysis, where level 1 is the
	 * molecular function, biological process, and cellular location level
	 */
	public int nmingolevel;

	/**
	 *boolean as to whether to include biological process annotations
	 */
	public boolean bpontoval;

	/**
	 *boolean as to whether to include cellular component annotations
	 */
	public boolean bcontoval;

	/**
	 *boolean as to whether to include molecular function annotations
	 */
	public boolean bfontoval;

	/**
	 *The evidence code input string consisting of evidence codes of
	 * annotations to filter, delimited by a comma, pipe, or semicolon
	 **/
	public String szevidenceval;

	/**
	 *The taxon input string consisting of taxons of which to restrict
	 * annotations delimited by a comma, pipe, or semicolon
	 */
	public String sztaxonval;

	/**
	 *boolean as to whether actual size based enrichment correction should be
	 * based on randomization (true) or Bonferroni false
	 */
	public boolean brandomgoval;

	/**
	 *List of extra probe ids to add to the base set from the szextraval file
	 */
	public ArrayList extraprobes;

	/**
	 *List of extra gene ids to add to the base set from the szextraval file
	 * i^th index of extra genes should correspond to i^th index of extraprobes
	 */
	public ArrayList extragenes;

	/**
	 *An array of profiles GO enrichment records ordered based on their
	 * enrichment to a GO category
	 */
	public ProfileGORankingRec[] tpgr;

	/**
	 *Arraylist of hash sets of GO categories of each unique gene
	 */
	public ArrayList algoAllbase;

	/**
	 *Arraylist of hash sets of GO categories of each unique gene
	 */
	public ArrayList algoAll;

	/**
	 *Array of hash sets of GO categories of each unique gene
	 */
	public HashSet[] goAll;

	/**
	 *An array of record of GO categories and their best actual and expected
	 * enrichment p-values for any profile based on last DataSet invoked with
	 */
	public RecIDpval2[] theRecIDpval;

	/**
	 *An array of record of GO categories and their best actual enrichment
	 * p-values for any cluster of profiles based on last DataSet invoked with
	 */
	public RecIDpval[] theClusterRecIDpval;

	/**
	 *This file has the gene annotations
	 */
	public String szGoFile;

	/**
	 *This file has the gene Hierarchy
	 */
	public String szGoCategoryFile;

	/**
	 * This is the output file for batch runs, null if not used
	 */
	public String szBatchGOoutput;

	/**
	 *A count of the total count of the number of genes in a set (dtotal) and
	 * the number selected based on the ordering criteria (dmatch)
	 */
	public static class RecCount {
		RecCount(double dmatch, double dtotal) {
			this.dmatch = dmatch;
			this.dtotal = dtotal;
		}

		public double dmatch;
		public double dtotal;
	}

	/**
	 *Procedure prints the level of every category in htAncestors
	 */
	public void printLevel() {
		Iterator enumels = htAncestors.keySet().iterator();

		while (enumels.hasNext()) {
			String szel = (String) enumels.next();
			System.out.println(szel + "\t" + getLevel(szel, new HashSet()));
		}
	}

	/**
	 *Returns a HashSet of GO IDs with level below nmingolevel Removes those
	 * IDs from htGO
	 */
	public HashSet levelGO() {
		Iterator enumels = htGO.keySet().iterator();
		HashSet removeSet = new HashSet();

		// iterating over all levels
		while (enumels.hasNext()) {
			String szel = (String) enumels.next();
			int nlevel = getLevel(szel, new HashSet());

			if (nlevel < nmingolevel) {
				removeSet.add(szel);
			}
		}

		// removing from hierarchy
		Iterator removeels = removeSet.iterator();
		while (removeels.hasNext()) {
			htGO.remove((String) removeels.next());
		}
		return removeSet;
	}

	/**
	 *Returns the level of the GO ID of szTerm level is defined in terms of
	 * maximum length to a root
	 */
	public int getLevel(String szTerm, HashSet pathvisited) {
		int nmaxlevel = 0;

		if (pathvisited.contains(szTerm)) {
			System.out
					.println("Warning -- getLevel detected cycle with the term "
							+ szTerm);
			return 1;
		} else {
			pathvisited.add(szTerm);
		}

		Rec r = (Rec) htGO.get(szTerm);
		if (r == null) {
			throw new IllegalArgumentException(
					"Dangling reference found in gene ontology with term ID "
							+ szTerm);
		} else {
			HashSet parents = (HashSet) r.parents;
			Iterator parentsitr = parents.iterator();
			while (parentsitr.hasNext()) {
				int nlevel;
				String szparent = (String) parentsitr.next();
				nlevel = getLevel(szparent, pathvisited);
				pathvisited.remove(szparent);
				nmaxlevel = Math.max(nmaxlevel, nlevel);
			}
			return (nmaxlevel + 1);
		}

	}

	/**
	 *Returns a HashSet of ancestors for the go ID szgo
	 */
	HashSet getAncestors(String szgo, HashSet pathvisited) {

		if (pathvisited.contains(szgo)) {
			System.out
					.println("Warning -- getAncestors detected cycle with the term "
							+ szgo);
			return new HashSet();
		} else {
			HashSet ancestors = (HashSet) htAncestors.get(szgo);
			// will simply return the contents in htAncestors unless it is empty
			if (ancestors == null) {
				pathvisited.add(szgo);
				ancestors = new HashSet();
				Rec r = (Rec) htGO.get(szgo);

				if (r != null) {
					// record is in the ontology
					HashSet parents = r.parents;
					if (parents == null) {
						// if record does not have parents, but its alternative
						// ID does
						// then use the alternative ID's parents
						String realid = (String) htAlt.get(szgo);
						if (realid != null) {
							parents = ((Rec) htGO.get(realid)).parents;
						}
					}

					if (parents != null) {
						// going to iterate through the parents adding, any of
						// their ancestors
						// to the ancestor set
						Iterator parentsitr = parents.iterator();

						while (parentsitr.hasNext()) {
							String szcurrparent = (String) parentsitr.next();

							ancestors.add(szcurrparent);
							// getting the ancestor set of the parent
							HashSet currAncestors = getAncestors(szcurrparent,
									pathvisited);
							pathvisited.remove(szcurrparent);
							// adding any ancestor not currently in the set
							ancestors.addAll(currAncestors);
						}
					}
				}
				// adds a mapping to the ancestor set
				htAncestors.put(szgo, ancestors);
			}
			return ancestors;
		}
	}

	/**
	 *Builds a cross reference mapping which as keys is alternative identifiers
	 * that map to a subset of genes in htgenes
	 */
	public void buildxref(HashSet htgenes, HashMap htxref)
			throws FileNotFoundException, IOException {

		if ((szxrefval != null) && (!szxrefval.equals(""))) {
			// we have cross references from which to build
			BufferedReader br;
			try {
				// first tries gzip format, if that fails then does regular
				br = new BufferedReader(new InputStreamReader(
						new GZIPInputStream(new FileInputStream(szxrefval))));
			} catch (IOException ex) {
				br = new BufferedReader(new FileReader(szxrefval));
			}

			String szLine;
			while ((szLine = br.readLine()) != null) {
				// htfoundgenes contains genes in the htgenes also on the line
				HashSet htfoundgenes = new HashSet(3);
				// htallsyn has all genes on the line
				HashSet htallsyn = new HashSet();
				// entries can be delimited by a tab, semicolon, pipe, comma, or
				// double quote
				StringTokenizer st = new StringTokenizer(szLine, "\"\t;|,");
				while (st.hasMoreTokens()) {
					String sztoken = st.nextToken().trim().toUpperCase(
							Locale.ENGLISH);

					// no cross references with spot IDs
					// if there was them this would be uncommented
					// String szprobetoken = (String)
					// htProbetoGene.get(sztoken);
					// if (szprobetoken != null)
					// {
					// sztoken = szprobetoken;
					// cross reference contains a probe using
					// }

					// adding to synonym list
					htallsyn.add(sztoken);

					if (htgenes.contains(sztoken)) {
						// storing known genes
						htfoundgenes.add(sztoken);
					}
				}

				if (!htfoundgenes.isEmpty()) {
					// foundgenes not empty implies htallsynitr not empty
					Iterator htallsynitr = htallsyn.iterator();
					while (htallsynitr.hasNext()) {
						String szsyn = (String) htallsynitr.next();
						HashSet hssyngenes = (HashSet) htxref.get(szsyn);

						if (hssyngenes == null) {
							hssyngenes = new HashSet();
						}
						// mapping synonym to all terms
						hssyngenes.addAll(htfoundgenes);
						htxref.put(szsyn, hssyngenes);
					}
				}
			}
			br.close();
		}
	}

	/**
	 *Constructs a GoAnnotation object. szGoFile has the annotations while
	 * szGoCategoryFile has the hierarchy. Assumption is made that the set of
	 * genes and spots are the same for all repeats thus only pass in baseGenes1
	 * and baseProbes1
	 */
	public GoAnnotations(String szorganismsourceval, String szxrefsourceval,
			String szxrefval, String szGoFile, String szGoCategoryFile,
			String[] baseGenes1, String[] baseProbes1, int nsamplespval,
			int nmingo, int nmingolevel, String szextraval,
			String szcategoryIDval, boolean bspotincluded,
			String szevidenceval, String sztaxonval, boolean bpontoval,
			boolean bcontoval, boolean bfontoval, boolean brandomgoval,
			String szBatchGOoutput) throws FileNotFoundException, IOException,
			IllegalArgumentException {
		this.szBatchGOoutput = szBatchGOoutput;
		this.szorganismsourceval = szorganismsourceval;
		this.szxrefsourceval = szxrefsourceval;
		this.szGoFile = szGoFile;
		this.szGoCategoryFile = szGoCategoryFile;
		this.szevidenceval = szevidenceval;
		this.sztaxonval = sztaxonval;
		this.bpontoval = bpontoval;
		this.bcontoval = bcontoval;
		this.bfontoval = bfontoval;
		this.szxrefval = szxrefval;
		this.szextraval = szextraval;
		this.brandomgoval = brandomgoval;
		this.nmingo = nmingo;
		this.nmingolevel = nmingolevel;
		this.szcategoryIDval = szcategoryIDval;
		this.nsamplespval = nsamplespval;
		htGoLabels = new HashMap();
		htFullCount = new HashMap();
		htGO = new HashMap(20000);
		htAlt = new HashMap();
		htAncestors = new HashMap();
		htGeneNames = new HashMap();
		extragenes = new ArrayList();
		extraprobes = new ArrayList();

		StringTokenizer st;

		if (!szGoFile.equalsIgnoreCase("")) {
			// builds a mapping of category IDs to names for non-offical GO
			// categories
			loadIDCategory();
			if (!szGoCategoryFile.equalsIgnoreCase("")) {
				// reads .obo file
				try {
					BufferedReader br2;
					try {
						br2 = new BufferedReader(new InputStreamReader(
								new GZIPInputStream(new FileInputStream(
										szGoCategoryFile))));
					} catch (IOException ex) {
						br2 = new BufferedReader(new FileReader(
								szGoCategoryFile));
					}

					HashSet parents = new HashSet();
					String sztermID = "";
					String sztermName = "";
					String szLine;

					boolean bopenterm = false;

					while ((szLine = br2.readLine()) != null) {
						int nlength = szLine.length();
						if ((nlength > 0)
								&& ((szLine.charAt(0) == '[') && (szLine
										.charAt(nlength - 1) == ']'))
								&& (bopenterm)) {
							// store the term just completed
							Rec r = (Rec) htGO.get(sztermID);
							if (r == null) {
								// haven't seen this ID before
								htGO
										.put(sztermID, new Rec(parents,
												sztermName));
							} else {
								// seen this ID need to update the record
								if (!sztermName.equals("")) {
									r.sztermName = sztermName;
								}
								r.parents.addAll(parents);
							}
							bopenterm = false;
						}

						if (szLine.equals("[Term]")) {
							parents = new HashSet();
							sztermID = "";
							sztermName = "";
							bopenterm = true;
						} else if ((nlength > 0)
								&& ((szLine.charAt(0) == '[') && (szLine
										.charAt(nlength - 1) == ']'))) {
							// not a term resets parents
							parents = new HashSet();
						} else {
							String szfield = szLine.trim();
							if (szfield.startsWith("id:")) {
								StringTokenizer st2 = new StringTokenizer(
										szfield.substring(3).trim(), " !\t");
								if (st2.hasMoreTokens()) {
									sztermID = (String) st2.nextToken();
								} else {
									throw new IllegalArgumentException(
											"An id tag has no value");
								}
							} else if (szfield.startsWith("name:")) {
								sztermName = szfield.substring(5).trim();
								if (sztermName.equals("")) {
									throw new IllegalArgumentException("term "
											+ sztermID
											+ " name tag has no value");
								}
							} else if (szfield.startsWith("is_a:")) {
								StringTokenizer st2 = new StringTokenizer(
										szfield.substring(5).trim(), " !\t");
								if (st2.hasMoreTokens()) {
									parents.add(st2.nextToken());
								} else {
									throw new IllegalArgumentException("term "
											+ sztermID
											+ " is_a tag has no value");
								}
							} else if (szfield.startsWith("relationship:")) {
								StringTokenizer st2 = new StringTokenizer(
										szfield.substring(13).trim(), " !\t");
								if (st2.hasMoreTokens()) {
									st2.nextToken();

									if (st2.hasMoreTokens()) {
										String sz = st2.nextToken();
										parents.add(sz);// st2.nextToken());
									} else {
										throw new IllegalArgumentException(
												"term "
														+ sztermID
														+ " relationship part_of tag has no value");
									}
								} else {
									throw new IllegalArgumentException("term "
											+ sztermID
											+ " relationship tag has no value");
								}
							} else if (szfield.startsWith("alt_id:")) {
								// assuming alternated id comes after orig id

								StringTokenizer st2 = new StringTokenizer(
										szfield.substring(7).trim(), " !\t");

								if (st2.hasMoreTokens()) {
									htAlt.put(st2.nextToken(), sztermID);
								} else {
									throw new IllegalArgumentException("term "
											+ sztermID
											+ " alt_id tag has no value");
								}
							}
						}
					}

					// store the last term
					if (bopenterm) {
						Rec r = (Rec) htGO.get(sztermID);
						if (r == null) {
							htGO.put(sztermID, new Rec(parents, sztermName));
						} else {
							if (!sztermName.equals("")) {
								r.sztermName = sztermName;
							}
							r.parents.addAll(parents);
						}
					}
					br2.close();
				} catch (FileNotFoundException ex) {
					ex.printStackTrace(System.out);
				}
			}
			HashSet removedterms = levelGO();

			// now going to parse annotation file
			BufferedReader br;
			try {
				br = new BufferedReader(new InputStreamReader(
						new GZIPInputStream(new FileInputStream(szGoFile))));
			} catch (IOException ex) {
				br = new BufferedReader(new FileReader(szGoFile));
			}

			String szLine;
			String szannotation;
			szLine = br.readLine();
			if (szLine == null) {
				throw new IllegalArgumentException("Annotation file is empty!");
			} else {
				while ((szLine.equals("")) || (szLine.startsWith("!"))) {
					// reads blank lines and those that start with !
					szLine = br.readLine();
					if (szLine == null) {
						throw new IllegalArgumentException(
								"Annotation file is empty!");
					}
				}
			}

			// build a probe ID to gene entry mapping
			// and the set of all genes in a gene listing
			HashMap htProbetoGene = new HashMap();
			HashSet htgenes = new HashSet();
			for (int nindex = 0; nindex < baseGenes1.length; nindex++) {
				if ((szBatchGOoutput == null)
						|| (!baseGenes1[nindex].startsWith("#"))) {
					addprobegenemap(baseProbes1[nindex], baseGenes1[nindex],
							htProbetoGene, htgenes);
				}
			}

			// also need to add to htProbetoGene and htgenes
			// the genes and probe to gene mappings in the extra gene file
			if ((szextraval != null) && (!szextraval.equals(""))) {
				BufferedReader brextra;
				try {
					brextra = new BufferedReader(
							new InputStreamReader(new GZIPInputStream(
									new FileInputStream(szextraval))));
				} catch (IOException ex) {
					brextra = new BufferedReader(new FileReader(szextraval));
				}

				brextra.readLine(); // gets rid of header line
				StringTokenizer stextraLine;
				String szextragene, szextraprobe, szextraLine;
				int ncount = 0;
				while ((szextraLine = brextra.readLine()) != null) {
					stextraLine = new StringTokenizer(szextraLine, "\t", true);
					if ((stextraLine.hasMoreTokens()) || (!bspotincluded)) {
						// if spot IDs included and no more tokens just assume
						// it was a blank line and ignore
						if (bspotincluded) {
							szextraprobe = stextraLine.nextToken().trim()
									.toUpperCase(Locale.ENGLISH);
							if (szextraprobe.equals("\t")) {
								throw new IllegalArgumentException(
										"Missing spot symbol in pre-filtered file");
							} else if (stextraLine.hasMoreTokens()) {
								// gets tab
								stextraLine.nextToken();
							}
						} else {
							// automatically generate the spot ID
							szextraprobe = "EID_" + ncount;
						}
						ncount++;
						szextragene = "0";
						if (stextraLine.hasMoreTokens()) {
							szextragene = stextraLine.nextToken();
						}

						if ((szextragene.equals("\t"))
								|| (szextragene.equals("0"))) {
							// gene name missing
							szextragene = "0 " + "(SPOT_" + szextraprobe + ")";
						} else {
							if (szextragene.charAt(0) == '\"') {
								// strip quotes
								szextragene = szextragene.substring(1);
							}

							if (szextragene.charAt(szextragene.length() - 1) == '\"') {
								// strip quotes
								szextragene = szextragene.substring(0,
										szextragene.length() - 1);
							}
						}

						szextragene = szextragene.trim().toUpperCase(
								Locale.ENGLISH);
						addprobegenemap(szextraprobe, szextragene,
								htProbetoGene, htgenes);
						extragenes.add(szextragene);
						extraprobes.add(szextraprobe);
					}
				}
				brextra.close();
			}

			HashMap htxref = new HashMap();
			buildxref(htgenes, htxref);

			st = new StringTokenizer(szLine, "\t", false);
			StringTokenizer stwtabs = new StringTokenizer(szLine, "\t", true);

			int ntabs = stwtabs.countTokens() - st.countTokens();
			if (ntabs >= 14) {
				addOfficialLine(stwtabs, removedterms, htgenes, htxref,
						htProbetoGene);
				while ((szLine = br.readLine()) != null) {
					if (!szLine.equals("")) {
						stwtabs = new StringTokenizer(szLine, "\t", true);
						addOfficialLine(stwtabs, removedterms, htgenes, htxref,
								htProbetoGene);
					}
				}
			} else {

				// we are in the simple two column format
				while ((szLine = br.readLine()) != null) {
					// skipping blank lines
					if (!szLine.equals("")) {
						st = new StringTokenizer(szLine, "\t");
						String szAnnotationWithQuotes = st.nextToken();
						if (!st.hasMoreTokens()) {
							throw new IllegalArgumentException("Line:\n"
									+ szLine + "\nis missing an argument");
						} else {
							String szGoPortion = st.nextToken();
							StringTokenizer stAnnotation = new StringTokenizer(
									szAnnotationWithQuotes, "\";|,");
							// thing being annotated
							while (stAnnotation.hasMoreElements()) {
								szannotation = stAnnotation.nextToken().trim();
								szannotation = szannotation
										.toUpperCase(Locale.ENGLISH);
								String szprobegene = (String) htProbetoGene
										.get(szannotation);
								if (szprobegene != null) {

									addgenesgo(szprobegene, szGoPortion,
											removedterms, htgenes, htxref);
								}
								addgenesgo(szannotation, szGoPortion,
										removedterms, htgenes, htxref);
							}
						}
					}
				}
			}
			br.close();

			// loading ancestors
			Iterator enumels = htGoLabels.values().iterator();
			while (enumels.hasNext()) {
				HashSet vgolist = (HashSet) enumels.next();
				Iterator vgolistitr = ((HashSet) vgolist.clone()).iterator();
				// iterating through the cloned list
				while (vgolistitr.hasNext()) {
					String szcurrgo = (String) vgolistitr.next();

					HashSet ancestors = getAncestors(szcurrgo, new HashSet());
					Iterator ancestorsitr = ancestors.iterator();
					while (ancestorsitr.hasNext()) {
						String szancestor = (String) ancestorsitr.next();
						if (!removedterms.contains(szancestor)) {
							vgolist.add(szancestor);
						}
					}
				}
			}
		}

		nmaxsize = 1;
		algoAll = new ArrayList();
		for (int nindex = 0; nindex < baseGenes1.length; nindex++) {
			if ((szBatchGOoutput == null)
					|| (!baseGenes1[nindex].startsWith("#"))) {
				loadID(baseGenes1[nindex]);
			}
		}

		int nextragenessize = extragenes.size();
		for (int nindex = 0; nindex < nextragenessize; nindex++) {
			loadID((String) extragenes.get(nindex));
		}

		nmaxsize++;

		goAll = new HashSet[algoAll.size()];
		int nsizealgoAll = algoAll.size();
		for (int nindex = 0; nindex < nsizealgoAll; nindex++) {
			goAll[nindex] = (HashSet) algoAll.get(nindex);
		}
	}

	/**
	 *Adds a mapping of probe symbols to gene IDs, where the probe symbols are
	 * the sub-entries while the gene entry is the full field to htProbetoGene.
	 * Adds to htgenes the set of all genes IDs in a field, and also the full
	 * field. Entries in a field are delimited by a pipe, semicolon, comma, or
	 * double quote.
	 */
	public void addprobegenemap(String szprobefull, String szgenefull,
			HashMap htProbetoGene, HashSet htgenes) {

		String szprobe, szgene;
		StringTokenizer stGene = new StringTokenizer(szgenefull, "|;,\"");

		// adds to the set of genes all delimited entries and the full gene
		while (stGene.hasMoreTokens()) {
			szgene = stGene.nextToken().trim();
			htgenes.add(szgene);
		}
		htgenes.add(szgenefull);

		StringTokenizer stProbe = new StringTokenizer(szprobefull, "|;,\"");
		while (stProbe.hasMoreTokens()) {
			// maps a probe id to its gene entry
			szprobe = stProbe.nextToken().trim();
			htProbetoGene.put(szprobe, szgenefull);
		}
	}

	/**
	 *Maps annotations to GO categories for offical 15 column GO annotation
	 * files. Maps entries in column 2, 3, 11, 12 to the GO category in column 5
	 * provided it meets the evidence code (column 7), aspect (column 9), and
	 * taxon (column 13) constraints
	 */
	public void addOfficialLine(StringTokenizer stwtabs, HashSet removedterms,
			HashSet htgenes, HashMap htxref, HashMap htProbetoGene) {
		String szDBobjectID, szDBobjectSymbol, szDBobjectName = null, szSynonym = null, szGOID;
		String sztoken;
		String szAspect, szEvidence, szTaxon;
		String szqualifier;

		// we've get an official GO Annotation file
		HashSet annotations;
		// 1. Discards DB
		stwtabs.nextToken();
		stwtabs.nextToken();

		szDBobjectID = stwtabs.nextToken().trim().toUpperCase(Locale.ENGLISH);
		// 2. DB_Object_ID
		stwtabs.nextToken();

		szDBobjectSymbol = stwtabs.nextToken().trim().toUpperCase(
				Locale.ENGLISH);
		// 3. DB_Object_Symbol
		stwtabs.nextToken();

		// 4. QUALIFIER
		szqualifier = stwtabs.nextToken();
		if (!szqualifier.equals("\t")) {
			stwtabs.nextToken();
		}

		szGOID = stwtabs.nextToken();// .trim().toUpperCase(Locale.ENGLISH);
		// 5. GOID
		stwtabs.nextToken();

		// 6. db ref
		stwtabs.nextToken();
		stwtabs.nextToken();

		// 7. evidence
		szEvidence = stwtabs.nextToken().trim().toUpperCase(Locale.ENGLISH);
		stwtabs.nextToken();

		// 8. with/from
		sztoken = stwtabs.nextToken();
		if (!sztoken.equals("\t")) {
			stwtabs.nextToken();
		}

		szAspect = stwtabs.nextToken().trim().toUpperCase(Locale.ENGLISH);
		// 9. Aspect
		stwtabs.nextToken();

		sztoken = stwtabs.nextToken();
		if (!sztoken.equals("\t")) {
			// 10. DB_Object_Name
			szDBobjectName = sztoken.trim().toUpperCase(Locale.ENGLISH);
			stwtabs.nextToken();
		}

		sztoken = stwtabs.nextToken();
		if (!sztoken.equals("\t")) {
			// 11. DB_Object_Synonym
			szSynonym = sztoken.trim().toUpperCase(Locale.ENGLISH);
			stwtabs.nextToken();
		}
		// System.out.println("**"+szSynonym);

		// 12. DB_Object_Type
		stwtabs.nextToken();
		stwtabs.nextToken();

		// 13. Taxon
		szTaxon = stwtabs.nextToken();

		if ((!szqualifier.contains("NOT"))
				&& ((((bcontoval) && szAspect.equals("C"))
						|| ((bpontoval) && szAspect.equals("P")) || ((bfontoval) && szAspect
						.equals("F")))
						&& ((szevidenceval.equals("")) || validEvidence(
								szevidenceval, szEvidence)) && ((sztaxonval
						.equals("")) || containsTaxon(sztaxonval, szTaxon)))) {
			// we've got a valid valid line

			HashSet gocategories;

			// first going to see if an annotation symbol matches a probe
			// if so then will annotate the gene match the probe to be of szGOID
			// otherwise will try seeing if the symbol is a valid gene
			// if also not a valid gene will see if it is a synonym for a valid
			// gene based on the cross references

			String szprobeDBobjectID = (String) htProbetoGene.get(szDBobjectID);
			if (szprobeDBobjectID != null) {
				gocategories = getgocategories(szprobeDBobjectID);
				addgenesgocategory(szGOID, removedterms, gocategories);
			} else if (htgenes.contains(szDBobjectID)) {
				gocategories = getgocategories(szDBobjectID);
				addgenesgocategory(szGOID, removedterms, gocategories);
			} else {
				HashSet hsgenes = (HashSet) htxref.get(szDBobjectID);
				if (hsgenes != null) {
					// synonym for valid gene(s) going to annotate these genes
					Iterator hsgenesitr = hsgenes.iterator();
					while (hsgenesitr.hasNext()) {
						String szalgene = (String) hsgenesitr.next();
						gocategories = getgocategories(szalgene);
						addgenesgocategory(szGOID, removedterms, gocategories);
					}
				}
			}

			String szprobeDBobjectSymbol = (String) htProbetoGene
					.get(szDBobjectSymbol);
			if (szprobeDBobjectSymbol != null) {
				gocategories = getgocategories(szprobeDBobjectSymbol);
				addgenesgocategory(szGOID, removedterms, gocategories);
			} else if (htgenes.contains(szDBobjectSymbol)) {
				gocategories = getgocategories(szDBobjectSymbol);
				addgenesgocategory(szGOID, removedterms, gocategories);
			} else {
				HashSet hsgenes = (HashSet) htxref.get(szDBobjectSymbol);
				if (hsgenes != null) {
					Iterator hsgenesitr = hsgenes.iterator();
					while (hsgenesitr.hasNext()) {
						String szalgene = (String) hsgenesitr.next();
						gocategories = getgocategories(szalgene);
						addgenesgocategory(szGOID, removedterms, gocategories);
					}
				}
			}

			// adding first part before underscore, if multiple tokens
			// for szDBobjectSymbol since the first part of maps to gene symbols
			// while the second part is the gene database
			StringTokenizer stu = new StringTokenizer(szDBobjectSymbol, "_");
			if (stu.countTokens() > 1) {
				String szprefix = stu.nextToken();
				String szprefixprobeDBobjectSymbol = (String) htProbetoGene
						.get(szprefix);
				if (szprefixprobeDBobjectSymbol != null) {
					gocategories = getgocategories(szprefixprobeDBobjectSymbol);
					addgenesgocategory(szGOID, removedterms, gocategories);
				} else if (htgenes.contains(szprefix)) {
					gocategories = getgocategories(szprefix);
					addgenesgocategory(szGOID, removedterms, gocategories);
				} else {
					HashSet hsgenes = (HashSet) htxref.get(szprefix);
					if (hsgenes != null) {
						Iterator hsgenesitr = hsgenes.iterator();
						while (hsgenesitr.hasNext()) {
							String szalgene = (String) hsgenesitr.next();
							gocategories = getgocategories(szalgene);
							addgenesgocategory(szGOID, removedterms,
									gocategories);
						}
					}
				}
			}

			if (szDBobjectName != null) {
				String szprobeDBobjectName = (String) htProbetoGene
						.get(szDBobjectName);
				if (szprobeDBobjectName != null) {
					gocategories = getgocategories(szprobeDBobjectName);
					addgenesgocategory(szGOID, removedterms, gocategories);
				} else if (htgenes.contains(szDBobjectName)) {
					gocategories = getgocategories(szDBobjectName);
					addgenesgocategory(szGOID, removedterms, gocategories);
				} else {
					HashSet hsgenes = (HashSet) htxref.get(szDBobjectName);
					if (hsgenes != null) {
						Iterator hsgenesitr = hsgenes.iterator();
						while (hsgenesitr.hasNext()) {
							String szalgene = (String) hsgenesitr.next();
							gocategories = getgocategories(szalgene);
							addgenesgocategory(szGOID, removedterms,
									gocategories);
						}
					}
				}
			}

			if (szSynonym != null) {
				// officially synonyms are delimited by |, but sometimes they
				// are also delimited by ; or ,
				// column delim temp
				StringTokenizer stSynonym = new StringTokenizer(szSynonym,
						":|;,\"");
				while (stSynonym.hasMoreTokens()) {

					String szterm = stSynonym.nextToken().trim();
					String szprobeterm = (String) htProbetoGene.get(szterm);

					if (szprobeterm != null) {
						gocategories = getgocategories(szprobeterm);
						addgenesgocategory(szGOID, removedterms, gocategories);
					} else if (htgenes.contains(szterm)) {
						gocategories = getgocategories(szterm);
						addgenesgocategory(szGOID, removedterms, gocategories);
					} else {
						HashSet hsgenes = (HashSet) htxref.get(szterm);
						if (hsgenes != null) {
							Iterator hsgenesitr = hsgenes.iterator();
							while (hsgenesitr.hasNext()) {
								String szalgene = (String) hsgenesitr.next();
								gocategories = getgocategories(szalgene);
								addgenesgocategory(szGOID, removedterms,
										gocategories);
							}
						}
					}
				}
			}
		}
	}

	/**
	 *Returns true if szEvidence is not in szevidenceval where the entries of
	 * szevidenceval are delimited by a semicolon, comma, or pipe
	 */
	public boolean validEvidence(String szevidenceval, String szEvidence) {
		StringTokenizer st = new StringTokenizer(szevidenceval, ";,|");
		while (st.hasMoreTokens()) {
			if (st.nextToken().equalsIgnoreCase(szEvidence)) {
				return false;
			}
		}
		return true;
	}

	/**
	 *Returns true if an entry in sztaxonval is also an entry in szTaxon or
	 * after prepending a 'taxon:' szTaxon entries are delimited by pipes
	 * sztaxonval entries are delimited by semicolons, commas, and pipes
	 */
	public boolean containsTaxon(String sztaxonval, String szTaxon) {
		StringTokenizer stTaxon = new StringTokenizer(szTaxon, "|");
		while (stTaxon.hasMoreTokens()) {
			String szTaxontoken = stTaxon.nextToken();
			StringTokenizer st = new StringTokenizer(sztaxonval, ";,|");
			while (st.hasMoreTokens()) {
				String sztaxonvaltoken = st.nextToken();

				if ((sztaxonvaltoken.equalsIgnoreCase(szTaxontoken))
						|| (("taxon:" + sztaxonvaltoken)
								.equalsIgnoreCase(szTaxontoken))) {
					return true;
				}
			}
		}
		return false;
	}

	/**
	 *Returns the HashSet of gocategories for szannotation If such a set does
	 * not exist creates it
	 */
	public HashSet getgocategories(String szannotation) {
		HashSet gocategories = (HashSet) htGoLabels.get(szannotation);
		if (gocategories == null) {
			gocategories = new HashSet();
			htGoLabels.put(szannotation, gocategories);
		}

		return gocategories;
	}

	// ////////////////////////////////////////////////////////////////////////
	/**
	 *Adds szcategory or its alternative ID to gocategories, provide it is not
	 * in removedterms also if this category has not been seen before add the ID
	 * and its name if available to htGO. Takes a single entry, use addgenesgo
	 * to annotate a gene to belong to multiple categories.
	 */
	public void addgenesgocategory(String szcategory, HashSet removedterms,
			HashSet gocategories) {
		// szcategory is assumed to be only one gocategory
		String szalt = (String) htAlt.get(szcategory);
		if (szalt != null) {
			szcategory = szalt;
		}

		if (!removedterms.contains(szcategory)) {
			if (gocategories.add(szcategory)) {
				// if this is a new GO category will wan
				if (htGO.get(szcategory) == null) {
					String szcategoryname = (String) htIDCategory
							.get(szcategory);
					if (szcategoryname == null) {
						// using ID for name since not found
						szcategoryname = szcategory;
					}
					htGO
							.put(szcategory, new Rec(new HashSet(),
									szcategoryname));
				}
			}
		}
	}

	// //////////////////////////////////////////////////////////////////////
	/**
	 *Maps ids to category names for non-official go categories specified in
	 * the file szcategoryIDval in the htIDCategory hash map.
	 */
	public void loadIDCategory() {
		htIDCategory = new HashMap();
		if (!szcategoryIDval.equals("")) {
			try {
				BufferedReader br;
				try {
					br = new BufferedReader(new InputStreamReader(
							new GZIPInputStream(new FileInputStream(
									szcategoryIDval))));
				} catch (IOException ex) {
					br = new BufferedReader(new FileReader(szcategoryIDval));
				}

				String szLine;
				while ((szLine = br.readLine()) != null) {
					StringTokenizer st = new StringTokenizer(szLine, "\t");
					if (st.countTokens() >= 2) {
						String szid = (String) st.nextToken();
						String szname = (String) st.nextToken();
						htIDCategory.put(szid, szname);
					}
				}
				br.close();
			} catch (FileNotFoundException fnfe) {
				System.out.println(szcategoryIDval + " not found");
			} catch (IOException ioex) {
				ioex.printStackTrace(System.out);
			}
		}
	}

	// ///////////////////////////////////////////////////////////////////////
	/**
	 *Add go annotations to szannotation or its cross-reference for every entry
	 * in szGoPortion delimited by tabs, semiclons, commas, pipes, or double
	 * quotes.
	 */
	public void addgenesgo(String szannotation, String szGoPortion,
			HashSet removedterms, HashSet htgenes, HashMap htxref) {
		// szgoportion needs to be tokenized
		if (htgenes.contains(szannotation)) {
			// we have the gene
			HashSet gocategories = getgocategories(szannotation);
			StringTokenizer stGO = new StringTokenizer(szGoPortion, "\t;,|\"");
			while (stGO.hasMoreTokens()) {
				String szcategory = stGO.nextToken().trim();
				addgenesgocategory(szcategory, removedterms, gocategories);
			}
		} else {
			// check cross-references
			HashSet hsgenes = (HashSet) htxref.get(szannotation);
			if (hsgenes != null) {
				Iterator hsgenesitr = hsgenes.iterator();
				while (hsgenesitr.hasNext()) {
					String szalgene = (String) hsgenesitr.next();
					HashSet gocategories = getgocategories(szalgene);
					StringTokenizer stGO = new StringTokenizer(szGoPortion,
							"\t;,|\"");

					while (stGO.hasMoreTokens()) {
						String szcategory = stGO.nextToken().trim();
						addgenesgocategory(szcategory, removedterms,
								gocategories);
					}
				}
			}
		}
	}

	/**
	 *Updates htBase for all categories of baseGene. Adds gocategories to
	 * algoAllbase
	 */
	public void loadIDbase(String baseGene, HashMap htBase,
			ArrayList algoAllbase) {

		HashSet currGo = labelsForID(baseGene); // gets the go categories for
												// the gene

		algoAllbase.add(currGo);

		Iterator currGoitr = currGo.iterator();
		while (currGoitr.hasNext()) {
			// iterating over all GO categories of the gene
			String szcategory = (String) currGoitr.next();
			Object objFull = htFullCount.get(szcategory);

			// updating the number of annotated genes of the category
			if (objFull == null) {
				// have not seen this category before
				htBase.put(szcategory, new Integer(1));
			} else {
				int ncurrval = ((Integer) objFull).intValue();
				ncurrval++;

				htBase.put(szcategory, new Integer(ncurrval));

			}
		}
	}

	// //////////////////////////////////////////////////////

	/**
	 *Adds genes to htGeneNames, updates the numtotalgenes count htFullCounts,
	 * and adds to algoAll
	 */
	public void loadID(String baseGene) {
		if (htGeneNames.get(baseGene) == null) {
			// we have not seen this gene before
			// setting to false on the define gene set list
			htGeneNames.put(baseGene, Boolean.valueOf(false));// new
																// Boolean(false));

			// adding ID
			numtotalgenes++; // this is now all non-duplicate and 0 genes
			HashSet currGo = labelsForID(baseGene); // gets the go categories
													// for the gene

			algoAll.add(currGo);

			Iterator currGoitr = currGo.iterator();
			while (currGoitr.hasNext()) {
				// iterating over all GO categories of the gene
				String szcategory = (String) currGoitr.next();
				Object objFull = htFullCount.get(szcategory);
				// updating the number of annotated genes of the category
				if (objFull == null) {
					// have not seen this category before
					htFullCount.put(szcategory, new Integer(1));
					numcategory++;
					if (nmingo <= 1) {
						nlegalgo++;
					}
				} else {
					int ncurrval = ((Integer) objFull).intValue();
					ncurrval++;
					if (ncurrval == nmingo) {
						// category meets the minimum GO size requirement
						nlegalgo++;
					}

					htFullCount.put(szcategory, new Integer(ncurrval));
					if (ncurrval > nmaxsize) {
						// maximum number of annotated genes of any category
						nmaxsize = ncurrval;
					}
				}
			}
		}
	}

	// //////////////////////////////////////////////////////
	/**
	 *Given a string of delimited IDs, returns any categories which map to an
	 * entry of the string or the whole string
	 */
	public HashSet labelsForID(String szID) {
		HashSet currGo = new HashSet();

		StringTokenizer st2 = new StringTokenizer(szID, ";|,");
		if (st2.countTokens() > 1) {
			// add the whole thing also, will add some parts in the while loop
			HashSet tempcurrGo = (HashSet) htGoLabels.get(szID);
			if (tempcurrGo != null) {
				Iterator tempcurrGoitr = tempcurrGo.iterator();
				while (tempcurrGoitr.hasNext()) {
					String szterm = (String) tempcurrGoitr.next();
					currGo.add(szterm);
				}
			}
		}

		while (st2.hasMoreTokens()) {
			String szidtoken = st2.nextToken().trim();
			HashSet tempcurrGo = (HashSet) htGoLabels.get(szidtoken);
			if (tempcurrGo != null) {
				Iterator tempcurrGoitr = tempcurrGo.iterator();
				while (tempcurrGoitr.hasNext()) {
					String szterm = (String) tempcurrGoitr.next();
					currGo.add(szterm);
				}
			}
		}

		return currGo;
	}

	// //////////////////////////////////////////////////////////////

	/**
	 *Returns true if the profiles are sorted based on a GO category or user
	 * defined gene set and szGene is part of that category or gene set.
	 */
	public boolean isOrder(String szGene) {
		if ((szsortcommand.equals("go")) || (szsortcommand.equals("expgo"))) {
			return geneIsOfType(szGene, szSelectedGO);
		} else if ((szsortcommand.equals("define"))
				|| (szsortcommand.equals("expdefine"))) {
			Boolean bquerygene = (Boolean) htGeneNames.get(szGene);
			if ((bquerygene != null) && (((Boolean) bquerygene).booleanValue())) {
				return true;
			}
		}
		return false;
	}

	// ////////////////////////////////////////////////////

	/**
	 *Returns true if the gene ID szID is of type szCategory any sub entry or
	 * the entire string could match where entries are delimited by semicolons,
	 * pipes, or commas
	 */
	public boolean geneIsOfType(String szID, String szCategory) {
		int nval = 0;

		StringTokenizer st2 = new StringTokenizer(szID, ";|,");

		if (st2.countTokens() > 1) {
			// try the whole string
			HashSet tempcurrGo = (HashSet) htGoLabels.get(szID);
			if (tempcurrGo != null) {
				if (tempcurrGo.contains(szCategory)) {
					return true;
				}
			}
		}

		while (st2.hasMoreTokens()) {
			HashSet tempcurrGo = (HashSet) htGoLabels.get(st2.nextToken()
					.trim());
			if (tempcurrGo != null) {
				if (tempcurrGo.contains(szCategory)) {
					return true;
				}
			}
		}
		return false;
	}

	// /////////////////////////////////////////////////////
	/**
	 *Stores GO enrichment results for one category
	 */
	public static class GoRec {
		/**
		 *The number of genes in the category and in the set of interest
		 */
		public double dcategoryselect;
		/**
		 *The number of genes in the category
		 */
		public double dcategoryall;
		/**
		 *The uncorrected actual size based p-value
		 */
		public double dpvalue;
		/**
		 *The uncorrected expected size based p-value
		 */
		public double dpvaluebinom;
		/**
		 *The ID of the category
		 */
		public String szgoid;
		/**
		 *The name of the category
		 */
		public String szgocategory;
		/**
		 *The corrected actual size based p-value
		 */
		public double dcorrectedpvalue;
		/**
		 *The corrected expected size based p-value
		 */
		public double dcorrectedpvaluebinom;

		/**
		 *Constructor without expected size based p-value
		 */
		public GoRec(double dcategoryselect, double dcategoryall,
				double dpvalue, String szgocategory, String szgoid) {
			this.dcategoryselect = dcategoryselect;
			this.dcategoryall = dcategoryall;
			this.dpvalue = dpvalue;
			this.szgoid = szgoid;
			this.szgocategory = szgocategory;
		}

		/**
		 *Constructor with expected size based p-value
		 */
		GoRec(double dcategoryselect, double dcategoryall, double dpvalue,
				double dpvaluebinom, String szgocategory, String szgoid) {
			this.dcategoryselect = dcategoryselect;
			this.dcategoryall = dcategoryall;
			this.dpvalue = dpvalue;
			this.dpvaluebinom = dpvaluebinom;
			this.szgoid = szgoid;
			this.szgocategory = szgocategory;
		}

		/**
		 *Prints the contents of all the variables in the record
		 */
		public void print() {
			System.out.print(szgoid + "\t" + szgocategory + "\t"
					+ dcategoryselect + "\t" + dcategoryall + "\t" + dpvalue
					+ "\t" + dcorrectedpvalue + "\t" + dpvaluebinom + "\t"
					+ dcorrectedpvaluebinom);
		}
	}

	/**
	 *Encapsulates a set of GO enrichment results
	 */
	public static class GoResults {
		/**
		 *The total number of genes in the base set
		 */
		public double dall;

		/**
		 *The number of genes in the set being analyzed
		 */
		public double dselect;

		/**
		 *An array of GO enrichment records
		 */
		public GoRec[] goRecArray;

		/**
		 *Constructor for GoResults
		 */
		public GoResults(double dall, double dselect, GoRec[] goRecArray) {
			this.dall = dall;
			this.dselect = dselect;
			this.goRecArray = goRecArray;
		}

		/**
		 *Prints out the contents of GoResults
		 */
		public void print(PrintWriter pw, boolean brandomgoval, String szHeader) {

			NumberFormat nf3 = NumberFormat.getInstance(Locale.ENGLISH);
			nf3.setMinimumFractionDigits(3);
			nf3.setMaximumFractionDigits(3);

			NumberFormat nf1 = NumberFormat.getInstance(Locale.ENGLISH);
			nf1.setMinimumFractionDigits(1);
			nf1.setMaximumFractionDigits(1);
			nf1.setGroupingUsed(false);

			pw
					.println("GeneSet\tCategoryID\tCategoryName\t#Genes Category\t#Genes assigned\t#Genes Expected\t#Genes Enriched\tp-value\tCorrected p-value\tfold\t#Genes in Set\t#Total Genes");
			for (int nindex = 0; nindex < goRecArray.length; nindex++) {
				double dexpected = dselect * goRecArray[nindex].dcategoryall
						/ dall;
				double denriched = goRecArray[nindex].dcategoryselect
						- dexpected;
				double dfold = (goRecArray[nindex].dcategoryselect / dselect)
						/ (goRecArray[nindex].dcategoryall / dall);

				String szcorrectedpval;
				if (goRecArray[nindex].dcorrectedpvalue < 0.001) {
					if (brandomgoval) {
						szcorrectedpval = "<0.001";
					} else {
						szcorrectedpval = Util
								.doubleToSz(goRecArray[nindex].dcorrectedpvalue);
					}
				} else {
					szcorrectedpval = nf3
							.format(goRecArray[nindex].dcorrectedpvalue);
				}

				String szexpected = nf1.format(dexpected);
				String szenriched = nf1.format(denriched);
				double dval = Double.parseDouble(szenriched);
				if (dval > 0) {
					szenriched = "+" + szenriched;
				} else if (denriched == 0) {
					szenriched = "0.0";
				}

				String szpvalue = Util.doubleToSz(goRecArray[nindex].dpvalue);
				pw.println(szHeader + "\t" + goRecArray[nindex].szgoid + "\t"
						+ goRecArray[nindex].szgocategory + "\t"
						+ goRecArray[nindex].dcategoryall + "\t"
						+ goRecArray[nindex].dcategoryselect + "\t"
						+ szexpected + "\t" + szenriched + "\t" + szpvalue
						+ "\t" + szcorrectedpval + "\t" + nf3.format(dfold)
						+ "\t" + dselect + "\t" + dall);

			}
		}
	}

	/**
	 *Stores a pair of p-values one is based on the hypergeometric and the
	 * other based on the binomial
	 */
	static class PvalRec {
		PvalRec(double dpvalhyper, double dpvalbinom) {
			this.dpvalhyper = dpvalhyper;
			this.dpvalbinom = dpvalbinom;
		}

		double dpvalhyper;
		double dpvalbinom;
	}

	// //////////////////////////////////////////////////////////////////////////////////
	/**
	 *Returns the GO results for actual size based enrichment for a set where
	 * the indicies of the set of genes in the set is ids If bqueryset is true,
	 * then only genes also in the query set will be used If htinames is
	 * non-null, then only genes also in htinames will be used
	 */
	public GoResults getCategory(ArrayList ids, String[] genenames,
			ArrayList[] assignments, boolean bqueryset, HashSet htinames)

	{
		return getCategory(ids, genenames, assignments, bqueryset, htinames, -1);
	}

	// /////////////////////////////////////////////////////////////////////////////////

	/**
	 *Returns the GO results for actual size based enrichment for a set where
	 * the set of genes in the set are those with positive weight
	 */
	public GoResults getCategory(String[] genenames, boolean[] binbase,
			boolean[] binset) {
		// contains a mapping from category ids to the number of genes in the
		// set belonging to that category
		HashMap htGoCounts = new HashMap();
		int nbaseselect = 0;
		double dselect = 0; // the number of genes in the set
		double dweight;

		HashMap htBaseCount = new HashMap();
		algoAllbase = new ArrayList();

		for (int ngeneindex = 0; ngeneindex < binset.length; ngeneindex++) {
			if (binbase[ngeneindex]) {
				String szGene = genenames[ngeneindex];
				int navgweight = 1;
				HashSet hsGo = labelsForID(szGene);
				algoAllbase.add(hsGo);
				nbaseselect += navgweight;
				if (binset[ngeneindex]) {
					dselect += navgweight;
				}

				Iterator hsGoitr = hsGo.iterator();
				while (hsGoitr.hasNext()) {
					String szCategory = (String) hsGoitr.next();

					Object objBase = htBaseCount.get(szCategory);
					// updating the number of annotated genes of the category
					if (objBase == null) {
						// have not seen this category before
						htBaseCount.put(szCategory, new Integer(1));
					} else {
						int ncurrval = ((Integer) objBase).intValue();
						ncurrval++;
						htBaseCount.put(szCategory, new Integer(ncurrval));
					}

					if (binset[ngeneindex]) {

						Object obj = htGoCounts.get(szCategory);
						if (obj == null) {
							htGoCounts.put(szCategory, new Double(navgweight));
						} else {
							double dcurrweight = ((Double) obj).doubleValue();
							htGoCounts.put(szCategory, new Double(dcurrweight
									+ navgweight));
						}
					}
				}
			}
		}

		return computePValuesBase(htGoCounts, htBaseCount, dselect,
				nbaseselect, binbase);
	}

	// ////////////////////////////////////////////////////////////////////////////////

	/**
	 *Returns a record with GoResults, included corrected and uncorrected
	 * p-values for actual size enrichment for the category counts in htGoCounts
	 */
	public GoResults computePValuesBase(HashMap htGoCounts,
			HashMap htBaseCount, double dselect, int nbaseselect,
			boolean[] binbase) {
		// dselect is sum of number selected
		HashMap htHyper = new HashMap();
		// maps the number of genes selected in category and total in category
		// to p-value
		// this cache of previously computed values speeds things up

		String szGoID;
		int numentry = 0;
		Iterator eGoCounts = htGoCounts.entrySet().iterator();
		while (eGoCounts.hasNext()) {
			double dcategoryselect = ((Double) ((Map.Entry) eGoCounts.next())
					.getValue()).doubleValue();
			if (dcategoryselect >= nmingo) {
				numentry++;
			}
		}

		eGoCounts = htGoCounts.entrySet().iterator();
		GoRec[] theGoRecA = new GoRec[numentry];
		int ncatindex = 0;

		int nrandom = (int) Math.ceil(dselect);
		while (eGoCounts.hasNext()) {
			Map.Entry themap = (Map.Entry) eGoCounts.next();
			double dcategoryselect = ((Double) themap.getValue()).doubleValue();

			if (dcategoryselect >= nmingo) {
				szGoID = (String) themap.getKey();
				int ncategoryall = ((Integer) htBaseCount.get(szGoID))
						.intValue();
				int nval = (int) Math.ceil(dcategoryselect - 1);
				String szpair = nval + "," + ncategoryall;
				Object pvalObj = htHyper.get(szpair);
				double dpval;
				if (pvalObj == null) {
					dpval = StatUtil.hypergeometrictail(nval, ncategoryall,
							nbaseselect - ncategoryall, nrandom);

					htHyper.put(szpair, new Double(dpval));
				} else {
					dpval = ((Double) pvalObj).doubleValue();
				}

				String szgoterm = ((Rec) htGO.get(szGoID)).sztermName;
				theGoRecA[ncatindex] = new GoRec(dcategoryselect, ncategoryall,
						dpval, szgoterm, szGoID);
				ncatindex++;
			}
		}

		// sorts the records based on uncorrected p-value
		Arrays.sort(theGoRecA, new RecCompare());

		if (brandomgoval) {

			HashSet[] goAllbase = new HashSet[nbaseselect];

			for (int nindex = 0; nindex < goAllbase.length; nindex++) {
				goAllbase[nindex] = (HashSet) algoAllbase.get(nindex);
			}

			// randomized multiple hypothesis correction
			// nrandom is the number of samples to draw
			int[] rindex = new int[nrandom];
			// rindex contains the random selected indicies

			// nsamplespval is the number of trials
			double[] dminpval = new double[nsamplespval];
			int[] ncount = new int[nrandom];
			// ncount contains the smallest category size for that gene set size

			for (int nsample = 0; nsample < nsamplespval; nsample++) {
				for (int nindex = 0; nindex < ncount.length; nindex++) {
					ncount[nindex] = nmaxsize + 1;
				}

				dminpval[nsample] = 1;

				for (int nindex = 0; nindex < nrandom; nindex++) {
					rindex[nindex] = nindex;
				}

				// drawing nrandom elements from a set of numtotalgenes elements
				// where each element is equally likely
				for (int nindex = nrandom; nindex < nbaseselect; nindex++) {
					if (Math.random() < ((double) nrandom / (double) (nindex + 1))) {
						rindex[(int) Math.floor(nrandom * Math.random())] = nindex;
					}
				}
				// random genes selected now going to score them
				HashSet hsGo;
				htGoCounts = new HashMap();
				for (int nindex = 0; nindex < nrandom; nindex++) {
					// going through each gene getting its GO results
					hsGo = goAllbase[rindex[nindex]];
					Iterator hsGoitr = hsGo.iterator();
					while (hsGoitr.hasNext()) {
						String szCategory = (String) hsGoitr.next();
						Object obj = htGoCounts.get(szCategory);
						if (obj == null) {
							htGoCounts.put(szCategory, new Integer(1));
						} else {
							int ncurrweight = ((Integer) obj).intValue();
							htGoCounts.put(szCategory, new Integer(
									ncurrweight + 1));
						}
					}
				}

				eGoCounts = htGoCounts.entrySet().iterator();

				while (eGoCounts.hasNext()) {
					Map.Entry themap = (Map.Entry) eGoCounts.next();
					szGoID = (String) themap.getKey();
					int ncategoryselect = ((Integer) themap.getValue())
							.intValue();
					int ncategoryall = ((Integer) htBaseCount.get(szGoID))
							.intValue();

					if (ncount[--ncategoryselect] > ncategoryall) {
						// found a smaller category with the same number
						// selected
						ncount[ncategoryselect] = ncategoryall;
					}
				}

				// trying to find the smallest p-value
				int nsmallest = nmaxsize + 1;
				for (int nindex = ncount.length - 1; nindex >= nmingo - 1; nindex--) {
					if (ncount[nindex] < nsmallest) {
						// to have a better p-value category size must be
						// smaller than for a larger
						// number of genes selected
						String szpair = nindex + "," + ncount[nindex];
						Object pvalObj = htHyper.get(szpair);
						double dpval;
						nsmallest = ncount[nindex];
						if (pvalObj == null) {
							dpval = StatUtil.hypergeometrictail(nindex,
									ncount[nindex], nbaseselect
											- ncount[nindex], nrandom);
							htHyper.put(szpair, new Double(dpval));
						} else {
							dpval = ((Double) pvalObj).doubleValue();
						}

						if (dpval < dminpval[nsample]) {
							dminpval[nsample] = dpval;
						}
					}
				}
			}

			Arrays.sort(dminpval);
			int npvalindex = 0;
			for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
				while ((npvalindex < dminpval.length)
						&& (dminpval[npvalindex] <= theGoRecA[nrecindex].dpvalue)) {
					npvalindex++;
				}

				theGoRecA[nrecindex].dcorrectedpvalue = (double) npvalindex
						/ (double) nsamplespval;
			}
		} else {
			// Bonferroni multiple hypothesis correction
			for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
				theGoRecA[nrecindex].dcorrectedpvalue = Math.min(
						theGoRecA[nrecindex].dpvalue * nlegalgo, 1);
			}
		}

		return new GoResults(nbaseselect, dselect, theGoRecA);
	}

	// //////////////////////////////////////////////////////////////////////////////////
	/**
	 *Returns the GO results for actual size based enrichment for a set where
	 * the set of genes in the set are those with positive weight
	 */
	public GoResults getCategory(String[] genenames, double[] weight) {
		// contains a mapping from category ids to the number of genes in the
		// set belonging to that category
		HashMap htGoCounts = new HashMap();

		double dselect = 0; // the number of genes in the set
		double dweight;

		for (int ngeneindex = 0; ngeneindex < weight.length; ngeneindex++) {
			if (weight[ngeneindex] > 0) {
				String szGene = genenames[ngeneindex];
				double davgweight = weight[ngeneindex];
				HashSet hsGo = labelsForID(szGene);

				dselect += davgweight;

				Iterator hsGoitr = hsGo.iterator();
				while (hsGoitr.hasNext()) {
					String szCategory = (String) hsGoitr.next();
					Object obj = htGoCounts.get(szCategory);
					if (obj == null) {
						htGoCounts.put(szCategory, new Double(davgweight));
					} else {
						double dcurrweight = ((Double) obj).doubleValue();
						htGoCounts.put(szCategory, new Double(dcurrweight
								+ davgweight));
					}
				}
			}
		}

		return computePValues(htGoCounts, dselect);
	}

	// //////////////////////////////////////////////////////////////////////////////////
	/**
	 *Returns the GO results for expected size based enrichment for a set where
	 * the indicies of the set of genes in the set is ids If dexpected is <0,
	 * returns an actual size based enrichment If bqueryset is true, then only
	 * genes also in the query set will be used If htinames is non-null, then
	 * only genes also in htinames will be used
	 */
	public GoResults getCategory(ArrayList ids, String[] genenames,
			ArrayList[] assignments, boolean bqueryset, HashSet htinames,
			double dexpected) {
		// contains a mapping from category ids to the number of genes in the
		// set belonging to that category
		HashMap htGoCounts = new HashMap();

		int nsize = ids.size();
		double dselect = 0; // the number of genes in the set
		double dweight;

		for (int nindex = 0; nindex < nsize; nindex++) {
			int ngeneindex = ((Integer) ids.get(nindex)).intValue();

			if (((!bqueryset) && (htinames == null))
					|| ((htinames != null) && (htinames
							.contains(genenames[ngeneindex])))
					|| ((bqueryset) && (isOrder(genenames[ngeneindex])))) {
				// valid gene
				String szGene = genenames[ngeneindex];
				double davgweight = 1 / (double) assignments[ngeneindex].size();
				HashSet hsGo = labelsForID(szGene);

				dselect += davgweight;

				Iterator hsGoitr = hsGo.iterator();
				while (hsGoitr.hasNext()) {
					String szCategory = (String) hsGoitr.next();
					Object obj = htGoCounts.get(szCategory);
					if (obj == null) {
						htGoCounts.put(szCategory, new Double(davgweight));
					} else {
						double dcurrweight = ((Double) obj).doubleValue();
						htGoCounts.put(szCategory, new Double(dcurrweight
								+ davgweight));
					}
				}
			}
		}

		if (dexpected < 0) {
			return computePValues(htGoCounts, dselect);
		} else {
			return computePValues(htGoCounts, dselect, dexpected);
		}
	}

	/**
	 *Comparator for GoRec based on actual size p-value
	 */
	public static class RecCompare implements Comparator {
		/**
		 *Compares first by dpvalue (lower comes first) and then by
		 * dcategoryselect (greater comes first)
		 */
		public int compare(Object o1, Object o2) {
			GoRec grec1 = (GoRec) o1;
			GoRec grec2 = (GoRec) o2;

			if (grec1.dpvalue < grec2.dpvalue)
				return -1;
			if (grec1.dpvalue > grec2.dpvalue)
				return 1;
			if (grec1.dcategoryselect > grec2.dcategoryselect)
				return -1;
			if (grec1.dcategoryselect < grec2.dcategoryselect)
				return 1;
			return 0;
		}
	}

	/**
	 *Comparator for GoRec based on expected size p-value
	 */
	public static class RecCompareBinom implements Comparator {
		/**
		 *Compares first by dpvaluebinom (lower comes first) and then by
		 * dcategoryselect (greater comes first)
		 */
		public int compare(Object o1, Object o2) {
			GoRec grec1 = (GoRec) o1;
			GoRec grec2 = (GoRec) o2;

			if (grec1.dpvaluebinom < grec2.dpvaluebinom)
				return -1;
			if (grec1.dpvaluebinom > grec2.dpvaluebinom)
				return 1;
			if (grec1.dcategoryselect > grec2.dcategoryselect)
				return -1;
			if (grec1.dcategoryselect < grec2.dcategoryselect)
				return 1;
			return 0;
		}
	}

	// //////////////////////////////////////////////////////////////////////////////////
	/**
	 *Returns a record with GoResults, included corrected and uncorrected
	 * p-values for actual size enrichment for the category counts in htGoCounts
	 */
	public GoResults computePValues(HashMap htGoCounts, double dselect) {
		// dselect is sum of number selected
		// long lstart = System.currentTimeMillis();
		HashMap htHyper = new HashMap();
		// maps the number of genes selected in category and total in category
		// to p-value
		// this cache of previously computed values speeds things up

		String szGoID;
		int numentry = 0;
		Iterator eGoCounts = htGoCounts.entrySet().iterator();
		while (eGoCounts.hasNext()) {
			double dcategoryselect = ((Double) ((Map.Entry) eGoCounts.next())
					.getValue()).doubleValue();
			if (dcategoryselect >= nmingo) {
				numentry++;
			}
		}

		eGoCounts = htGoCounts.entrySet().iterator();
		GoRec[] theGoRecA = new GoRec[numentry];
		int ncatindex = 0;

		int nrandom = (int) Math.ceil(dselect);
		while (eGoCounts.hasNext()) {
			Map.Entry themap = (Map.Entry) eGoCounts.next();
			double dcategoryselect = ((Double) themap.getValue()).doubleValue();

			if (dcategoryselect >= nmingo) {
				szGoID = (String) themap.getKey();
				int ncategoryall = ((Integer) htFullCount.get(szGoID))
						.intValue();
				int nval = (int) Math.ceil(dcategoryselect - 1);
				String szpair = nval + "," + ncategoryall;
				Object pvalObj = htHyper.get(szpair);
				double dpval;
				if (pvalObj == null) {
					dpval = StatUtil.hypergeometrictail(nval, ncategoryall,
							numtotalgenes - ncategoryall, nrandom);
					htHyper.put(szpair, new Double(dpval));
				} else {
					dpval = ((Double) pvalObj).doubleValue();
				}

				String szgoterm = ((Rec) htGO.get(szGoID)).sztermName;
				theGoRecA[ncatindex] = new GoRec(dcategoryselect, ncategoryall,
						dpval, szgoterm, szGoID);
				ncatindex++;
			}
		}

		// sorts the records based on uncorrected p-value
		Arrays.sort(theGoRecA, new RecCompare());

		if (brandomgoval) {
			// randomized multiple hypothesis correction
			// nrandom is the number of samples to draw
			int[] rindex = new int[nrandom];
			// rindex contains the random selected indicies

			// nsamplespval is the number of trials
			double[] dminpval = new double[nsamplespval];
			int[] ncount = new int[nrandom];
			// ncount contains the smallest category size for that gene set size

			for (int nsample = 0; nsample < nsamplespval; nsample++) {
				for (int nindex = 0; nindex < ncount.length; nindex++) {
					ncount[nindex] = nmaxsize + 1;
				}

				dminpval[nsample] = 1;

				for (int nindex = 0; nindex < nrandom; nindex++) {
					rindex[nindex] = nindex;
				}

				// drawing nrandom elements from a set of numtotalgenes elements
				// where each element is equally likely
				for (int nindex = nrandom; nindex < numtotalgenes; nindex++) {
					if (Math.random() < ((double) nrandom / (double) (nindex + 1))) {
						rindex[(int) Math.floor(nrandom * Math.random())] = nindex;
					}
				}
				// random genes selected now going to score them
				HashSet hsGo;
				htGoCounts = new HashMap();
				for (int nindex = 0; nindex < nrandom; nindex++) {
					// going through each gene getting its GO results
					hsGo = goAll[rindex[nindex]];
					Iterator hsGoitr = hsGo.iterator();
					while (hsGoitr.hasNext()) {
						String szCategory = (String) hsGoitr.next();
						Object obj = htGoCounts.get(szCategory);
						if (obj == null) {
							htGoCounts.put(szCategory, new Integer(1));
						} else {
							int ncurrweight = ((Integer) obj).intValue();
							htGoCounts.put(szCategory, new Integer(
									ncurrweight + 1));
						}
					}
				}

				eGoCounts = htGoCounts.entrySet().iterator();

				while (eGoCounts.hasNext()) {
					Map.Entry themap = (Map.Entry) eGoCounts.next();
					szGoID = (String) themap.getKey();
					int ncategoryselect = ((Integer) themap.getValue())
							.intValue();
					int ncategoryall = ((Integer) htFullCount.get(szGoID))
							.intValue();

					if (ncount[--ncategoryselect] > ncategoryall) {
						// found a smaller category with the same number
						// selected
						ncount[ncategoryselect] = ncategoryall;
					}
				}

				// trying to find the smallest p-value
				int nsmallest = nmaxsize + 1;
				for (int nindex = ncount.length - 1; nindex >= nmingo - 1; nindex--) {
					if (ncount[nindex] < nsmallest) {
						// to have a better p-value category size must be
						// smaller than for a larger
						// number of genes selected
						String szpair = nindex + "," + ncount[nindex];
						Object pvalObj = htHyper.get(szpair);
						double dpval;
						nsmallest = ncount[nindex];
						if (pvalObj == null) {
							dpval = StatUtil.hypergeometrictail(nindex,
									ncount[nindex], numtotalgenes
											- ncount[nindex], nrandom);
							htHyper.put(szpair, new Double(dpval));
						} else {
							dpval = ((Double) pvalObj).doubleValue();
						}

						if (dpval < dminpval[nsample]) {
							dminpval[nsample] = dpval;
						}
					}
				}
			}

			Arrays.sort(dminpval);
			int npvalindex = 0;
			for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
				while ((npvalindex < dminpval.length)
						&& (dminpval[npvalindex] <= theGoRecA[nrecindex].dpvalue)) {
					npvalindex++;
				}
				theGoRecA[nrecindex].dcorrectedpvalue = (double) npvalindex
						/ (double) nsamplespval;

			}
		} else {
			// Bonferroni multiple hypothesis correction
			for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
				theGoRecA[nrecindex].dcorrectedpvalue = Math.min(
						theGoRecA[nrecindex].dpvalue * nlegalgo, 1);
			}
		}

		return new GoResults(numtotalgenes, dselect, theGoRecA);
	}

	// //////////////////////////////////////////////////////////////////////////
	/**
	 *Returns a record with GoResults, included corrected and uncorrected
	 * p-values for expected size enrichment for the category counts in
	 * htGoCounts
	 */
	public GoResults computePValues(HashMap htGoCounts, double dselect,
			double dexpect) {
		// dselect is sum of number selected
		HashMap htHyperBinom = new HashMap();

		String szGoID;
		double dnullprob = dexpect / numtotalgenes;

		int numentry = 0;
		Iterator eGoCounts = htGoCounts.entrySet().iterator();
		while (eGoCounts.hasNext()) {
			// counting the number of valid go entries (having more than mingo)
			double dcategoryselect = ((Double) ((Map.Entry) eGoCounts.next())
					.getValue()).doubleValue();
			if (dcategoryselect >= nmingo) {
				numentry++;
			}
		}
		// ////////////////////////////////////////////////////////////////////////

		eGoCounts = htGoCounts.entrySet().iterator();
		GoRec[] theGoRecA = new GoRec[numentry];
		int ncatindex = 0;
		int nrandom = (int) Math.ceil(dselect);

		while (eGoCounts.hasNext()) {
			Map.Entry themap = (Map.Entry) eGoCounts.next();
			double dcategoryselect = ((Double) themap.getValue()).doubleValue();

			if (dcategoryselect >= nmingo) {
				szGoID = (String) themap.getKey();
				int ncategoryall = ((Integer) htFullCount.get(szGoID))
						.intValue();
				int nval = (int) Math.ceil(dcategoryselect - 1);
				String szpair = nval + "," + ncategoryall;
				Object pvalObj = htHyperBinom.get(szpair);
				double dpvalhyper, dpvalbinom;
				if (pvalObj == null) {
					dpvalhyper = StatUtil
							.hypergeometrictail(nval, ncategoryall,
									numtotalgenes - ncategoryall, nrandom);

					dpvalbinom = StatUtil.binomialtail(nval, ncategoryall,
							dnullprob);

					htHyperBinom.put(szpair,
							new PvalRec(dpvalhyper, dpvalbinom));
				} else {
					dpvalhyper = ((PvalRec) pvalObj).dpvalhyper;
					dpvalbinom = ((PvalRec) pvalObj).dpvalbinom;
				}

				String szgoterm = ((Rec) htGO.get(szGoID)).sztermName;

				theGoRecA[ncatindex] = new GoRec(dcategoryselect, ncategoryall,
						dpvalhyper, dpvalbinom, szgoterm, szGoID);
				ncatindex++;
			}
		}

		if (brandomgoval) {
			int[] rindex = new int[nrandom];
			double[] dminpvalhyper = new double[nsamplespval];
			int[] ncount = new int[nrandom];
			for (int nsample = 0; nsample < nsamplespval; nsample++) {
				for (int nindex = 0; nindex < ncount.length; nindex++) {
					ncount[nindex] = nmaxsize + 1;
				}

				dminpvalhyper[nsample] = 1;

				for (int nindex = 0; nindex < nrandom; nindex++) {
					rindex[nindex] = nindex;
				}

				// drawing nrandom elements from a set of numtotalgenes elements
				// where each element is equally likely
				for (int nindex = nrandom; nindex < numtotalgenes; nindex++) {
					if (Math.random() < ((double) nrandom / (double) (nindex + 1))) {
						rindex[(int) Math.floor(nrandom * Math.random())] = nindex;
					}
				}

				HashSet hsGo;
				htGoCounts = new HashMap();
				for (int nindex = 0; nindex < nrandom; nindex++) {
					hsGo = goAll[rindex[nindex]];

					Iterator hsGoitr = hsGo.iterator();
					while (hsGoitr.hasNext()) {
						String szCategory = (String) hsGoitr.next();
						Object obj = htGoCounts.get(szCategory);
						if (obj == null) {
							htGoCounts.put(szCategory, new Integer(1));
						} else {
							int ncurrweight = ((Integer) obj).intValue();
							htGoCounts.put(szCategory, new Integer(
									ncurrweight + 1));
						}
					}
				}

				eGoCounts = htGoCounts.entrySet().iterator();

				while (eGoCounts.hasNext()) {
					Map.Entry themap = (Map.Entry) eGoCounts.next();
					szGoID = (String) themap.getKey();
					int ncategoryselect = ((Integer) themap.getValue())
							.intValue();
					int ncategoryall = ((Integer) htFullCount.get(szGoID))
							.intValue();

					if (ncount[--ncategoryselect] > ncategoryall) {
						ncount[ncategoryselect] = ncategoryall;
					}
				}

				int nsmallest = nmaxsize + 1;
				for (int nindex = ncount.length - 1; nindex >= nmingo - 1; nindex--) {
					if (ncount[nindex] < nsmallest) {
						String szpair = nindex + "," + ncount[nindex];
						Object pvalObj = htHyperBinom.get(szpair);
						// double dpvalbinom = -1;
						double dpvalhyper;
						nsmallest = ncount[nindex];
						if (pvalObj == null) {
							dpvalhyper = StatUtil.hypergeometrictail(nindex,
									ncount[nindex], numtotalgenes
											- ncount[nindex], nrandom);

							htHyperBinom.put(szpair,
									new PvalRec(dpvalhyper, -1));

						} else {
							dpvalhyper = ((PvalRec) pvalObj).dpvalhyper;
						}

						if (dpvalhyper < dminpvalhyper[nsample]) {
							dminpvalhyper[nsample] = dpvalhyper;
						}
					}
				}
			}

			Arrays.sort(dminpvalhyper);
			int npvalhyperindex = 0;

			for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
				while ((npvalhyperindex < dminpvalhyper.length)
						&& (dminpvalhyper[npvalhyperindex] <= theGoRecA[nrecindex].dpvalue)) {
					npvalhyperindex++;
				}
				theGoRecA[nrecindex].dcorrectedpvalue = (double) npvalhyperindex
						/ (double) nsamplespval;
			}
		} else {
			for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
				theGoRecA[nrecindex].dcorrectedpvalue = Math.min(
						theGoRecA[nrecindex].dpvalue * nlegalgo, 1);
			}
		}

		Arrays.sort(theGoRecA, new RecCompareBinom());
		for (int nrecindex = 0; nrecindex < theGoRecA.length; nrecindex++) {
			theGoRecA[nrecindex].dcorrectedpvaluebinom = Math.min(
					theGoRecA[nrecindex].dpvaluebinom * nlegalgo, 1);
		}

		return new GoResults(numtotalgenes, dselect, theGoRecA);
	}

	// //////////////////////////////////////////////////////////////////////////////////
	/**
	 *Stores info about a profile's ranking according to the current order by
	 * criteria
	 */
	public static class ProfileGORankingRec {
		/**
		 *profile id
		 */
		public int nprofile;
		/**
		 *Cluster profile belongs to, non-significant profiles have the
		 * greatest cluster value
		 */
		public int ncluster;
		/**
		 *profile enrichment p-value
		 */
		public double dpval;
		/**
		 *cluster enrichment p-value
		 */
		public double dclusterpval;
		/**
		 *Number of genes that are selected and belong to the profile
		 */
		public double dgenes;
		/**
		 *Number of genes that belong to the profile; max that could be
		 * selected in profile
		 */
		public double dmaxselect;
		/**
		 *Number of genes that are selected and belong to the cluster
		 */
		public double dgenescluster;
		/**
		 *Number of genes that belong to the cluster; max that could be
		 * selected in cluster
		 */
		public double dmaxselectcluster;
		/**
		 *Number of genes that are selected
		 */
		public double dgenestotal;
		/**
		 *Total number of genes; max that could be selected
		 */
		public double dmaxselecttotal;

		public ProfileGORankingRec(int nprofile, int ncluster, double dpval,
				double dclusterpval, double dgenes, double dmaxselect,
				double dgenescluster, double dmaxselectcluster,
				double dgenestotal, double dmaxselecttotal) {
			this.nprofile = nprofile;
			this.ncluster = ncluster;
			this.dpval = dpval;
			this.dclusterpval = dclusterpval;
			this.dgenes = dgenes;
			this.dmaxselect = dmaxselect;
			this.dgenescluster = dgenescluster;
			this.dmaxselectcluster = dmaxselectcluster;
			this.dgenestotal = dgenestotal;
			this.dmaxselecttotal = dmaxselecttotal;
		}
	}

	/**
	 *Compartor for two profile ranking records
	 */
	public class GORankingComparator implements Comparator {
		public int compare(Object o1, Object o2) {
			ProfileGORankingRec rec1 = (ProfileGORankingRec) o1;
			ProfileGORankingRec rec2 = (ProfileGORankingRec) o2;

			if (bcluster) {
				if (rec1.dclusterpval < rec2.dclusterpval)
					return -1;
				if (rec1.dclusterpval > rec2.dclusterpval)
					return 1;
				if (rec1.ncluster < rec2.ncluster)
					return -1;
				if (rec1.ncluster > rec2.ncluster)
					return 1;
				if (rec1.dpval < rec2.dpval)
					return -1;
				if (rec1.dpval > rec2.dpval)
					return 1;
				if (rec1.dgenes > rec2.dgenes)
					return -1;
				if (rec1.dgenes < rec2.dgenes)
					return 1;
				if (rec1.nprofile < rec2.nprofile)
					return -1;
				if (rec1.nprofile > rec2.nprofile)
					return 1;
			} else {
				if (rec1.dpval < rec2.dpval)
					return -1;
				if (rec1.dpval > rec2.dpval)
					return 1;
				if (rec1.ncluster < rec2.ncluster)
					return -1;
				if (rec1.ncluster > rec2.ncluster)
					return 1;
				if (rec1.dgenes > rec2.dgenes)
					return -1;
				if (rec1.dgenes < rec2.dgenes)
					return 1;
				if (rec1.nprofile < rec2.nprofile)
					return -1;
				if (rec1.nprofile > rec2.nprofile)
					return 1;
			}
			return 0;
		}
	}

	// ///////////////////////////////////////////////////////////////////////////
	/**
	 *Counts the total weight of assignments in assignedgenes. Counts the total
	 * weight of assignments of genes is assignedgenes that also meet the
	 * ordering criteria.
	 */
	protected RecCount incrementSelectData(ArrayList assignedgenes,
			String[] genesprobes, ArrayList[] assignments,
			boolean breallyquery, String szRankBy) {
		int npasize = assignedgenes.size();
		double dmatch = 0;
		double dtotal = 0;
		for (int nindex = 0; nindex < npasize; nindex++) {
			int ngeneindex = ((Integer) assignedgenes.get(nindex)).intValue();
			String szGene = genesprobes[ngeneindex];
			double davgweight = 1 / (double) assignments[ngeneindex].size();
			dtotal += davgweight;
			if (breallyquery) {
				Boolean bquerygene = (Boolean) htGeneNames.get(szGene);
				if ((bquerygene != null)
						&& (((Boolean) bquerygene).booleanValue())) {
					dmatch += davgweight;
				}
			} else {
				if (geneIsOfType(szGene, szRankBy)) {
					dmatch += davgweight;
				}
			}
		}
		// System.out.println(dmatch+" "+dtotal);
		return new RecCount(dmatch, dtotal);
	}

	// /////////////////////////////////////////////////////////////

	/**
	 *A record containing a GO ID and a p-val
	 */
	public static class RecIDpval {
		public double dpval;
		public String szid;

		RecIDpval(double dpval, String szid) {
			this.dpval = dpval;
			this.szid = szid;
		}
	}

	/**
	 *A record containing a GO ID and an actual and an expected size p-val
	 */
	public static class RecIDpval2 {
		public double dpvalbinom;
		public double dpvalhyper;
		public String szid;

		RecIDpval2(double dpvalhyper, double dpvalbinom, String szid) {
			this.dpvalhyper = dpvalhyper;
			this.dpvalbinom = dpvalbinom;
			this.szid = szid;
		}
	}

	/**
	 *Compares two recIDs first by p-value, then if a tie with a string
	 * comparison wiht a string comparison on the ID
	 */
	static class RecIDpvalCompare implements Comparator {
		public int compare(Object o1, Object o2) {
			RecIDpval ro1 = (RecIDpval) o1;
			RecIDpval ro2 = (RecIDpval) o2;

			if (ro1.dpval < ro2.dpval) {
				return -1;
			} else if (ro1.dpval > ro2.dpval) {
				return 1;
			} else {
				return ro1.szid.compareTo(ro2.szid);
			}
		}
	}

	/**
	 *Compares two recIDs first by actual size p-value, then expected size
	 * p-value, then if a tie with a string comparison on the ID
	 */
	static class RecIDpval2Compare implements Comparator {
		public int compare(Object o1, Object o2) {
			RecIDpval2 ro1 = (RecIDpval2) o1;
			RecIDpval2 ro2 = (RecIDpval2) o2;

			if (ro1.dpvalhyper < ro2.dpvalhyper) {
				return -1;
			} else if (ro1.dpvalhyper > ro2.dpvalhyper) {
				return 1;
			} else if (ro1.dpvalbinom < ro2.dpvalbinom) { //
				return -1;
			} else if (ro1.dpvalbinom > ro2.dpvalbinom) {
				return 1;
			} else {
				return ro1.szid.compareTo(ro2.szid);
			}
		}
	}

	// /////////////////////////////////////////////////////////////
	/**
	 *Iterates over hIDpval storing its contents in currRecIDA, and then
	 * sorting currRecIDA. Assumed the data entry of hIDpval is a single double.
	 */
	protected void hashtoList1(HashMap hIDpval, RecIDpval[] currRecIDA) {
		Set entrySet = hIDpval.entrySet();
		Iterator entryIterator = entrySet.iterator();
		int nsize = entrySet.size();

		for (int nindex = 0; nindex < nsize; nindex++) {
			Map.Entry themap = (Map.Entry) entryIterator.next();
			String szid = (String) themap.getKey();
			double dpval = ((Double) themap.getValue()).doubleValue();
			currRecIDA[nindex] = new RecIDpval(dpval, szid);
		}
		Arrays.sort(currRecIDA, new RecIDpvalCompare());
	}

	// /////////////////////////////////////////////////////////////

	/**
	 *Iterates over hIDpval storing its contents in currRecIDA, and then
	 * sorting currRecIDA. Assumed the data entry of hIDpval is a PvalRec.
	 */
	protected void hashtoList2(HashMap hIDpval, RecIDpval2[] currRecIDA) {
		Set entrySet = hIDpval.entrySet();
		Iterator entryIterator = entrySet.iterator();
		int nsize = entrySet.size();

		for (int nindex = 0; nindex < nsize; nindex++) {
			Map.Entry themap = (Map.Entry) entryIterator.next();
			String szid = (String) themap.getKey();
			PvalRec theRec = (PvalRec) themap.getValue();
			currRecIDA[nindex] = new RecIDpval2(theRec.dpvalhyper,
					theRec.dpvalbinom, szid);
		}
		Arrays.sort(currRecIDA, new RecIDpval2Compare());
	}

	// ///////////////////////////////////////////////////////////////
	/**
	 *Iterates over all the entries in htGoCounts seeing if the current
	 * enrichment beats the enrichment in hIDpval and if so updates it
	 */
	protected void updateBest(HashMap htGoCounts, HashMap hIDpval,
			int numtotalgenes, HashMap htFullCount, double dselect) {

		Set entrySet = htGoCounts.entrySet();
		Iterator entryIterator = entrySet.iterator();
		int nsize = entrySet.size();
		int nceildselect = (int) Math.ceil(dselect);

		for (int nindex = 0; nindex < nsize; nindex++) {
			Map.Entry themap = (Map.Entry) entryIterator.next();
			String szgoid = (String) themap.getKey();
			double dcategoryselect = ((Double) themap.getValue()).doubleValue();
			Double dpvalbest = (Double) hIDpval.get(szgoid);
			int ncategoryall = ((Integer) htFullCount.get(szgoid)).intValue();

			double dpvalcurr = StatUtil.hypergeometrictail((int) Math
					.ceil(dcategoryselect - 1), ncategoryall, numtotalgenes
					- ncategoryall, nceildselect);

			if (dpvalbest == null) {
				if (dpvalcurr <= 1) {
					// changed above to be <= 1
					hIDpval.put(szgoid, new Double(dpvalcurr));
				}
			} else if (dpvalcurr < dpvalbest.doubleValue()) {
				hIDpval.put(szgoid, new Double(dpvalcurr));
			}
		}
	}

	// ///////////////////////////////////////////////////////////////
	/**
	 *Iterates over all the entries in htGoCounts seeing if the current
	 * enrichment beats the enrichment in hIDpval for either the actual size
	 * enrichment or expected size enrichment and if so updates it
	 */
	protected void updateBestHyperBinom(HashMap htGoCounts, HashMap hIDpval,
			int numtotalgenes, HashMap htFullCount, double dselect,
			double dnumexpected) {
		Set entrySet = htGoCounts.entrySet();
		Iterator entryIterator = entrySet.iterator();
		int nsize = entrySet.size();
		double dprobprofile = dnumexpected / numtotalgenes;
		int nceildselect = (int) Math.ceil(dselect);

		for (int nindex = 0; nindex < nsize; nindex++) {
			Map.Entry themap = (Map.Entry) entryIterator.next();
			String szgoid = (String) themap.getKey();
			double dcategoryselect = ((Double) themap.getValue()).doubleValue();
			PvalRec dpvalbestRec = (PvalRec) hIDpval.get(szgoid);
			int ncategoryall = ((Integer) htFullCount.get(szgoid)).intValue();
			int nval = (int) Math.ceil(dcategoryselect - 1);

			double dpvalcurrhyper = StatUtil.hypergeometrictail(nval,
					ncategoryall, numtotalgenes - ncategoryall, nceildselect);

			double dpvalcurrbinom = StatUtil.binomialtail(nval, ncategoryall,
					dprobprofile);

			if (dpvalbestRec == null) {
				if ((dpvalcurrhyper <= 1) || (dpvalcurrbinom <= 1)) {
					hIDpval.put(szgoid, new PvalRec(dpvalcurrhyper,
							dpvalcurrbinom));
				}
			} else if ((dpvalcurrhyper < dpvalbestRec.dpvalhyper)
					|| (dpvalcurrbinom < dpvalbestRec.dpvalbinom)) {
				hIDpval.put(szgoid, new PvalRec(Math.min(dpvalcurrhyper,
						dpvalbestRec.dpvalhyper), Math.min(dpvalcurrbinom,
						dpvalbestRec.dpvalbinom)));
			}
		}
	}

	// ////////////////////////////////////////////////////////////////////////

	public void clusterFileResults(String szClusterFile) throws Exception {
		double dselect = 0;
		HashMap htGoCounts = new HashMap();
		FileReader frclusters = new FileReader(szClusterFile);
		BufferedReader brclusters = new BufferedReader(frclusters);
		GoResults theGoResults;
		String szLine;
		boolean bfirst = true;
		double dtotal = 0;
		int ncluster = 1;
		HashSet hsseen = new HashSet();

		PrintWriter pwBatchGOoutfile = new PrintWriter(new FileWriter(
				szBatchGOoutput));
		String szHeader = "";
		while ((szLine = brclusters.readLine()) != null) {
			// System.out.println(szLine);
			if ((szLine.startsWith("#")) || (bfirst)) {

				if (!bfirst) {
					theGoResults = computePValues(htGoCounts, dselect);
					// System.out.print(dtotal+"\t");
					theGoResults
							.print(pwBatchGOoutfile, brandomgoval, szHeader);
					dselect = 0;
					htGoCounts = new HashMap();
					hsseen = new HashSet();
					dtotal = 0;
				} else {
					bfirst = false;
				}
				pwBatchGOoutfile.println(szLine);// "Cluster\t"+ncluster+"\t");
				szHeader = szLine;
				ncluster++;
			} else {

				StringTokenizer st = new StringTokenizer(szLine, " \t");
				if (st.hasMoreTokens()) {
					String szGene = st.nextToken().trim().toUpperCase(
							Locale.ENGLISH);
					if (!hsseen.contains(szGene)) {

						hsseen.add(szGene);
						dtotal++;
						HashSet alGo = labelsForID(szGene);

						dselect++;

						Iterator alGoitr = alGo.iterator();

						while (alGoitr.hasNext()) {
							String szCategory = (String) alGoitr.next();
							Object obj = htGoCounts.get(szCategory);
							if (obj == null) {
								htGoCounts.put(szCategory, new Double(1));
							} else {
								double dcurrweight = ((Double) obj)
										.doubleValue();
								htGoCounts.put(szCategory, new Double(
										dcurrweight + 1));
							}
						}
					}
				}
			}
		}
		brclusters.close();
		theGoResults = computePValues(htGoCounts, dselect);
		theGoResults.print(pwBatchGOoutfile, brandomgoval, szHeader);

		pwBatchGOoutfile.close();
	}

}
