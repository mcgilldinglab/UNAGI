package edu.cmu.cs.sb.core;

import java.io.*;
import java.util.*;
import java.util.zip.*;

/**
 *The class encapsulates a set of gene expression data
 */
public class DataSetCore {
    
	/**
	 *The designated main data file associated with the set, others are repeats
	 */
	public String szInputFile;

	/**
	 *If repeat data comes from distinct time series (true) that is a
	 * longitudal time series, otherwise it is false and each column of the same
	 * time point between data sets is interchangeable
	 */
	public boolean bfullrepeat;

	/**
	 *The names of the other repeat files
	 */
	public String[] otherInputFiles;

	/**
	 *Contains genes filtered. Maps these gene names to the list of probe IDs
	 * associated with it
	 */
	public HashMap<String,String> htFiltered;

	/**
	 *The maximum number of missing values to prevent a gene from being
	 * filtered.
	 */
	public int nmaxmissing;

	/**
	 *Minimum average pairwise correlation a gene must have between full
	 * repeats if bfullrepeat is true
	 */
	public double dmincorrelation;

	/**
	 *Number of rows in the data matrix. This corresponds to the number of
	 * genes.
	 */
	public int numrows;

	/**
	 *Number of columns in the data matrix. This corresponds to the number of
	 * time points
	 */
	public int numcols;

	/**
	 *The expression data, row are genes, columns are time points in
	 * experiments
	 */
	public double[][] data;

	/**
	 *0 if data value is missing non-zero if present
	 */
	public int[][] pmavalues;

	/**
	 *True if the spot column was included in the data file
	 */
	public boolean bspotincluded;

	/**
	 *Present/missing data for one data set First dimension is gene Second
	 * dimension is spot Third is present/missing value
	 */
	public int[][][] genespottimepma = null;

	/**
	 *Present/missing data for several data sets First dimension is gene Second
	 * dimension is repeat Third dimension is spot Fourth dimension is
	 * present/missing value
	 */
	public int[][][][] generepeatspottimepma = null;

	/**
	 *Present/missing data for one data set First dimension is gene Second
	 * dimension is spot Third is expression value (in log ratio form against
	 * time zero)
	 */
	public double[][][] genespottimedata = null;

	/**
	 *Present/missing data for several data sets First dimension is gene Second
	 * dimension is repeat Third dimension is spot Fourth dimension is
	 * expression value (in log ratio form against time zero)
	 */
	public double[][][][] generepeatspottimedata = null;

	/**
	 *If the log-ratio taken (true); else data is already in log space (false)
	 */
	public boolean btakelog;

	/**
	 *The list of probe IDs in the current data set
	 */
	public String[] probenames;

	/**
	 *The list of gene names for the current data set
	 */
	public String[] genenames;

	/**
	 *The distribution of all the average pairwise correlations of genes across
	 * full repeats
	 */
	public double[] sortedcorrvals = null;

	/**
	 *The time points at which the expression data was sampled
	 */
	public String[] dsamplemins;

	/**
	 *The threshold value for required change
	 */
	public double dthresholdvalue = -1;

	/**
	 * True if gene change threshold for filtering is based on the max-min
	 * difference False if the gene change threshold for filtering is based on
	 * the absolute difference
	 */
	public boolean bmaxminval;

	/**
	 * True if a column of inital 0's should be added to the data file, false
	 * otherwise
	 */
	public boolean badd0 = false;

	/**
	 *The header string for the spot ID column
	 */
	public String szProbeHeader;

	/**
	 *The header string for the gene name column
	 */
	public String szGeneHeader;

	/**
	 * Empty constructor
	 */
	public DataSetCore() {

	}

	/**
	 *Constructor for individual variables
	 */
	private DataSetCore(String szInputFile, double[][][] genespottimedata,
			double[][][][] generepeatspottimedata, int[][][] genespottimepma,
			int[][][][] generepeatspottimepma, double[][] data,
			int[][] pmavalues, String[] probenames, String[] genenames,
			double dthresholdvalue, boolean btakelog, boolean bspotincluded,
			boolean badd0, String[] dsamplemins, double[] sortedcorrvals,
			int nmaxmissing, double dmincorrelation, HashMap<String,String> htFiltered,
			String szProbeHeader, String szGeneHeader, boolean bmaxminval,
			String[] otherInputFiles, boolean bfullrepeat) {
		this.otherInputFiles = otherInputFiles;
		this.bfullrepeat = bfullrepeat;
		this.szInputFile = szInputFile;
		this.bmaxminval = bmaxminval;
		this.htFiltered = htFiltered;
		this.szProbeHeader = szProbeHeader;
		this.szGeneHeader = szGeneHeader;
		this.dmincorrelation = dmincorrelation;
		this.nmaxmissing = nmaxmissing;
		this.genespottimedata = genespottimedata;
		this.generepeatspottimedata = generepeatspottimedata;
		this.genespottimepma = genespottimepma;
		this.generepeatspottimepma = generepeatspottimepma;
		this.data = data;
		this.pmavalues = pmavalues;
		this.probenames = probenames;
		this.genenames = genenames;
		this.dthresholdvalue = dthresholdvalue;
		this.btakelog = btakelog;
		this.bspotincluded = bspotincluded;
		this.badd0 = badd0;
		numrows = data.length;
		numcols = data[0].length;
		this.dsamplemins = dsamplemins;
		this.sortedcorrvals = sortedcorrvals;
	}

	// //////////////////////////////////////////////////////////////////
	/**
	 *Constructor copies each field
	 */
	public DataSetCore(DataSetCore theDataSetCore) {
		this.szInputFile = theDataSetCore.szInputFile;
		this.bfullrepeat = theDataSetCore.bfullrepeat;
		this.otherInputFiles = theDataSetCore.otherInputFiles;
		this.htFiltered = theDataSetCore.htFiltered;
		this.nmaxmissing = theDataSetCore.nmaxmissing;
		this.bmaxminval = theDataSetCore.bmaxminval;
		this.dmincorrelation = theDataSetCore.dmincorrelation;
		this.numrows = theDataSetCore.numrows;
		this.numcols = theDataSetCore.numcols;
		this.data = theDataSetCore.data;
		this.bspotincluded = theDataSetCore.bspotincluded;
		this.genespottimepma = theDataSetCore.genespottimepma;
		this.genespottimedata = theDataSetCore.genespottimedata;
		this.generepeatspottimepma = theDataSetCore.generepeatspottimepma;
		this.generepeatspottimedata = theDataSetCore.generepeatspottimedata;
		this.pmavalues = theDataSetCore.pmavalues;
		this.btakelog = theDataSetCore.btakelog;
		this.dthresholdvalue = theDataSetCore.dthresholdvalue;
		this.probenames = theDataSetCore.probenames;
		this.genenames = theDataSetCore.genenames;
		this.sortedcorrvals = theDataSetCore.sortedcorrvals;
		this.dsamplemins = theDataSetCore.dsamplemins;
		this.badd0 = theDataSetCore.badd0;
		this.szProbeHeader = theDataSetCore.szProbeHeader;
		this.szGeneHeader = theDataSetCore.szGeneHeader;
	}

	// //////////////////////////////////////////////////////////////////////////

	/**
	 *Add those genes from tga.extragenes that were filtered to thtFiltered
	 */
	public void addExtraToFilter(GoAnnotations tga) {

		// first stores those genes not filtered
		HashSet<String> htNotFiltered = new HashSet<String>();
		if (genenames != null) {
			for (int nrow = 0; nrow < genenames.length; nrow++) {
				htNotFiltered.add(genenames[nrow]);
			}
		}

		int nextrasize = tga.extragenes.size();
		for (int nrow = 0; nrow < nextrasize; nrow++) {
			String szextragene = (String) tga.extragenes.get(nrow);
			if (!htNotFiltered.contains(szextragene)) {
				// gene did not pass filter
				String szProbe = (String) htFiltered.get(szextragene);
				String szExtraProbe = (String) tga.extraprobes.get(nrow);
				if (szProbe == null) {
					// have not seen this gene yet adding its probe
					htFiltered.put(szextragene, szExtraProbe);
				} else {
					// the pre-filtered file may contain the same spot ID as a
					// gene that was filtered
					// in which case we do not want to list the spot ID twice
					StringTokenizer st = new StringTokenizer(szProbe, ";");
					boolean bfound = false;
					while ((st.hasMoreTokens()) && (!bfound)) {
						if (szExtraProbe.equals((String) st.nextToken())) {
							bfound = true;
						}
					}
					if (!bfound) {
						// probe is new adding it to the list of probes for the
						// gene
						htFiltered.put(szextragene, szProbe + ";"
								+ szExtraProbe);
					}
				}
			}
		}
	}

	// /////////////////////////////////////////////////////////////////////////////////////////
	/**
	 *Reads in the datafile stored in szInputFile
	 */
	protected void dataSetReader(String szInputFile, int nmaxmissing,
			double dthresholdvalue, double dmincorrelation, boolean btakelog,
			boolean bspotincluded, boolean brepeatset, boolean badd0)
			throws IOException, FileNotFoundException, IllegalArgumentException {
		htFiltered = new HashMap<String,String>();
		if (szInputFile.equals("")) {
			throw new IllegalArgumentException("No input file specified!");
		}

		BufferedReader brInputFile;

		// first tries reading GZIPInputStream format
		try {
			brInputFile = new BufferedReader(new InputStreamReader(
					new GZIPInputStream(new FileInputStream(szInputFile))));
		} catch (IOException ex) {
			brInputFile = new BufferedReader(new FileReader(szInputFile));
		}

		String szLine, szToken;
		StringTokenizer st, st2;
		String szHeaderLine;
		szHeaderLine = brInputFile.readLine();
		if (szHeaderLine == null) {
			throw new IllegalArgumentException("Input File " + szInputFile
					+ " is empty!");
		} else {
			while (szHeaderLine.equals("")) {
				szHeaderLine = brInputFile.readLine();
				if (szHeaderLine == null) {
					throw new IllegalArgumentException("Input File "
							+ szInputFile + " is empty!");
				}
			}
		}

		// checks if every col ends with a tab then we assume we have one less
		// column
		boolean balltabend = true;

		// stores the expression data into alInputFile
		ArrayList<String> alInputFile = new ArrayList<String>();
		while ((szLine = brInputFile.readLine()) != null) {
			StringTokenizer szblank = new StringTokenizer(szLine, " \t");

			if ((szblank.countTokens() >= 1) || (!bspotincluded)) {
				alInputFile.add(szLine);
				if (balltabend) {
					balltabend = (szLine.endsWith("\t"));
				}
			}
		}
		brInputFile.close();
		st = new StringTokenizer(szHeaderLine, "\t");
		st2 = new StringTokenizer(szHeaderLine, "\t", true);

		numcols = st2.countTokens() - st.countTokens();
		if (balltabend) {
			numcols--;
		}

		if (bspotincluded) {
			numcols--;
		}

		if (badd0) {
			numcols++;
		}

		dsamplemins = new String[Math.max(numcols, 1)];

		if (bspotincluded) {
			szProbeHeader = st2.nextToken();
			if (!szProbeHeader.equals("\t")) {
				if (!st2.hasMoreTokens()) {
					String szmsg = "Missing gene header.";
					if (numcols == -1) {
						szmsg += "\nConsider unchecking 'Spot IDs included in the data file'";
					}
					throw new IllegalArgumentException(szmsg);
				}
				st2.nextToken();
			} else {
				szProbeHeader = "";
			}
		} else {
			szProbeHeader = "SPOT";
		}

		szGeneHeader = st2.nextToken(); // gene header
		if (!szGeneHeader.equals("\t")) {
			if (st2.hasMoreTokens()) {
				st2.nextToken(); // flush tab
			}
		} else {
			szGeneHeader = ""; // no gene header given
		}

		int npoint;
		if (badd0) {
			dsamplemins[0] = "0";
			npoint = 1;
		} else {
			npoint = 0;
		}

		for (; npoint < numcols; npoint++) {
			if (st2.hasMoreTokens()) {
				szToken = st2.nextToken();
			} else {
				throw new IllegalArgumentException("Missing a column header");
			}

			if (!szToken.equals("\t")) {
				if (st2.hasMoreTokens()) {
					st2.nextToken();
				}
			} else {
				szToken = "";
			}
			dsamplemins[npoint] = szToken;
		}

		numrows = alInputFile.size();
		if (numrows == 0) {
			throw new IllegalArgumentException(szInputFile + " is empty!");
		}

		data = new double[numrows][numcols];
		pmavalues = new int[numrows][numcols];
		probenames = new String[numrows];
		genenames = new String[numrows];
		String sztoken;
		HashMap htProbeIDs = new HashMap();

		for (int nrow = 0; nrow < numrows; nrow++) {
			szLine = (String) alInputFile.get(nrow);

			st = new StringTokenizer(szLine, "\t", true);
			if (bspotincluded) {
				sztoken = st.nextToken();
				// blank probes not allowed, nor are duplicates
				if (sztoken.equals("\t")) {
					String szmsg;
					if (brepeatset) {
						szmsg = "Missing a Spot Name in the repeat/comparison set";
					} else {
						szmsg = "Missing a Spot Name";
						szmsg += "\nConsider unchecking 'Spot IDs included in the data file'";
					}
					throw new IllegalArgumentException(szmsg);
				} else {
					if (htProbeIDs.containsKey(sztoken)) {
						String szmsg;
						if (brepeatset) {
							szmsg = "Spot name " + sztoken
									+ " in repeat/comparison is not unique";
						} else {
							szmsg = "Spot name "
									+ sztoken
									+ " is not unique"
									+ "\nConsider unchecking 'Spot IDs included in the data file'";
						}

						throw new IllegalArgumentException(szmsg);
					} else {
						htProbeIDs.put(sztoken, null);
						probenames[nrow] = sztoken.trim().toUpperCase(
								Locale.ENGLISH);
					}

					if (st.hasMoreTokens()) {
						st.nextToken(); // flush token
					}
				}
			} else {
				probenames[nrow] = "ID_" + nrow;
			}

			if (!st.hasMoreTokens()) {
				genenames[nrow] = "0 " + "(SPOT_" + probenames[nrow] + ")";
				for (int ncol = 0; ncol < numcols; ncol++) {
					data[nrow][ncol] = 0;
					pmavalues[nrow][ncol] = 0;
				}
			} else {
				sztoken = st.nextToken();
				if ((sztoken.equals("\t")) || (sztoken.equals("0"))) {
					// gene name missing; 0 counts as missing name field
					if ((sztoken.equals("0")) && (st.hasMoreTokens())) {
						st.nextToken();
					}
					sztoken = "0 " + "(SPOT_" + probenames[nrow] + ")";
				} else {
					if ((sztoken.charAt(0) == '\"')
							&& (sztoken.charAt(sztoken.length() - 1) == '\"')) {
						// strings quotes
						sztoken = sztoken.substring(1, sztoken.length() - 1);
					}
					if (st.hasMoreTokens()) {
						st.nextToken(); // get rid of tab
					}
				}

				genenames[nrow] = sztoken.trim().toUpperCase(Locale.ENGLISH);

				boolean beol = false;
				int ncol;
				if (badd0) {
					data[nrow][0] = 0;
					pmavalues[nrow][0] = 2;
					ncol = 1;
				} else {
					ncol = 0;
				}

				for (; ncol < numcols; ncol++) {
					if (!st.hasMoreTokens()) {
						beol = true;
					} else {
						sztoken = st.nextToken();
					}

					if ((beol) || (sztoken.equals("\t"))) {
						data[nrow][ncol] = 0;
						pmavalues[nrow][ncol] = 0;
					} else {
						try {
							data[nrow][ncol] = Double.parseDouble(sztoken);
						} catch (NumberFormatException pe) {
							String szmsg = sztoken
									+ " is not a valid real number";
							if (brepeatset) {
								szmsg = "In the repeat/comparison set " + szmsg;
							} else if (((ncol == 0) || ((ncol == 1) && (badd0)))
									&& (!bspotincluded)) {
								szmsg += "\n"
										+ "Consider checking 'Spot IDs included in the data file'";
							}

							throw new IllegalArgumentException(szmsg);
						}

						if ((btakelog) && (data[nrow][ncol] <= 0)) {
							// negative values are counting as missing when not
							// already in log-space
							pmavalues[nrow][ncol] = 0;
						} else {
							pmavalues[nrow][ncol] = 2;
						}

						if (st.hasMoreTokens()) {
							sztoken = st.nextToken();
							if (sztoken.equals("\n")) {
								beol = true;
							}
						}
					}
				}
			}
		}
	}

	// //////////////////////////////////////////////////////////////
	/**
	 *Filters those rows that have a missing value at the first time point
	 */
	public DataSetCore filterMissing1point() {
		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		for (int nrow = 0; nrow < numrows; nrow++) {
			if (pmavalues[nrow][0] != 0) {
				goodrow[nrow] = true;
				ngoodrows++;
			} else {
				goodrow[nrow] = false;
			}
		}
		return filtergenesgeneral(goodrow, ngoodrows, true);
	}

	// //////////////////////////////////////////////////////////////
	/**
	 *Filter those rows which have nmaxmissing or more missing values or are
	 * missing the first time point value
	 */
	public DataSetCore filterMissing() {
		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		for (int nrow = 0; nrow < numrows; nrow++) {
			int nmissingcount = 0;
			int ncol = 0;
			boolean bgoodrow = true;

			if (pmavalues[nrow][0] == 0) {
				bgoodrow = false;
			}

			while ((bgoodrow) && (ncol < numcols)) {
				if (pmavalues[nrow][ncol] == 0) {
					nmissingcount++;
				}
				bgoodrow = (nmissingcount <= nmaxmissing);
				ncol++;
			}

			if (bgoodrow) {
				goodrow[nrow] = true;
				ngoodrows++;
			} else {
				goodrow[nrow] = false;
			}
		}

		return filtergenesgeneral(goodrow, ngoodrows, true);
	}

	// ///////////////////////////////////////////////////////////////
	/**
	 *Removes those rows which are the duplicate of another row. Stores in
	 * htgenenames for each gene name the index of all rows associated with it
	 */
	public DataSetCore filterDuplicates() {
		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		Hashtable<String,ArrayList<Integer>> htgenenames = 
			new Hashtable<String,ArrayList<Integer>>();
		// built hashtable of name to index
		for (int nrow = 0; nrow < numrows; nrow++) {
			ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
			if (indicies == null) {
				goodrow[nrow] = true;
				// this id is new
				indicies = new ArrayList<Integer>();
				ngoodrows++;
				htgenenames.put(genenames[nrow], indicies);
			} else {
				goodrow[nrow] = false;
				// already seen this id
				indicies.add(new Integer(nrow));
			}
		}

		return filtergenesgeneral(goodrow, ngoodrows, false);
	}

	// ///////////////////////////////////////////////////////////////////////
	/**
	 *Removes duplicate gene rows in the data file and combines there values
	 * using the median
	 */
	public DataSetCore averageAndFilterDuplicates() {

		boolean[] goodrow = new boolean[numrows];
		int ngoodrows = 0;

		Hashtable<String,ArrayList<Integer>> htgenenames = 
			new Hashtable<String,ArrayList<Integer>>();
		int nmaxsize = 1;
		// built hashtable of name to index
		for (int nrow = 0; nrow < numrows; nrow++) {
			goodrow[nrow] = true;
			ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
			if (indicies == null) {
				// this id is new
				indicies = new ArrayList<Integer>();
				ngoodrows++;
				htgenenames.put(genenames[nrow], indicies);
			} else {
				// already seen this id
				indicies.add(new Integer(nrow));
				if (indicies.size() > nmaxsize) {
					nmaxsize = indicies.size();
				}
			}
		}

		// going to store the values here then sort
		double[] vals = new double[nmaxsize + 1];

		int ngoodindex = -1;
		// store ratio and expression and pma values before merging genes
		genespottimedata = new double[ngoodrows][][];
		genespottimepma = new int[ngoodrows][][];

		for (int nrow = 0; nrow < numrows; nrow++) {
			if (goodrow[nrow]) {
				ngoodindex++;
				ArrayList<Integer> indicies = htgenenames.get(genenames[nrow]);
				int nindiciessize = indicies.size();

				genespottimedata[ngoodindex] = new double[nindiciessize + 1][];
				genespottimedata[ngoodindex][0] = data[nrow];
				genespottimepma[ngoodindex] = new int[nindiciessize + 1][];
				genespottimepma[ngoodindex][0] = pmavalues[nrow];
				// other rows may match this one
				for (int ncol = 0; ncol < numcols; ncol++) {
					// go through each column finding median
					int nvalindex = 0;
					int npma = 0;
					if (pmavalues[nrow][ncol] != 0) {
						vals[nvalindex] = data[nrow][ncol];
						nvalindex++;
						npma = pmavalues[nrow][ncol];
					}

					for (int nindex2 = 0; nindex2 < nindiciessize; nindex2++) {
						// go through all matches to this one
						int nrow2 = ((Integer) indicies.get(nindex2))
								.intValue();

						if (ncol == 0) {
							genespottimedata[ngoodindex][nindex2 + 1] = data[nrow2];
							genespottimepma[ngoodindex][nindex2 + 1] = pmavalues[nrow2];
							goodrow[nrow2] = false;
							probenames[nrow] += ";" + probenames[nrow2];
						}

						if (pmavalues[nrow2][ncol] != 0) {
							// this is a valid data value
							vals[nvalindex] = data[nrow2][ncol];
							nvalindex++;
							npma = Math.max(npma, pmavalues[nrow2][ncol]);
						}
					}

					pmavalues[nrow][ncol] = npma;
					if (nvalindex > 0) {
						// averages using median
						data[nrow][ncol] = Util.getmedian(vals, nvalindex);
					}
				}
			}
		}
		return filtergenesgeneral(goodrow, ngoodrows, false);
	}

	// ///////////////////////////////////////////////////////////////////////////
	/**
	 *Converts data into log-ratio versus the first time point. If btakelog is
	 * true then this the log base 2 of a value over the time point 0 value. If
	 * it is false then this is the difference with the time point 0 value.
	 */
        
	public DataSetCore logratio2() {
		double dnormval;
		double DLOG2 = Math.log(2);
                double pseudocount=1.0;
                
		for (int nrow = 0; nrow < numrows; nrow++) {
			dnormval = data[nrow][0];
			data[nrow][0] = 0;

			if (pmavalues[nrow][0] == 0) {
				for (int ncol = 1; ncol < numcols; ncol++) {
					//data[nrow][ncol] = Double.POSITIVE_INFINITY;
					data[nrow][ncol]=Math.log(data[nrow][ncol]+pseudocount)/DLOG2 - Math.log(dnormval+pseudocount)/DLOG2;
                                        pmavalues[nrow][ncol] = 0;
				}
			} else {
				for (int ncol = 1; ncol < numcols; ncol++) {
					if (btakelog) {
						/*
                                                data[nrow][ncol] = Math
							.log(data[nrow][ncol] / dnormval) / DLOG2;
                                                 */
                                               
                                                data[nrow][ncol]=Math.log(data[nrow][ncol]+pseudocount)/DLOG2 - Math.log(dnormval+pseudocount)/DLOG2;
					} else {
						data[nrow][ncol] = data[nrow][ncol] - dnormval;
					}
				}
			}
		}

		return new DataSetCore(szInputFile, genespottimedata,
				generepeatspottimedata, genespottimepma, generepeatspottimepma,
				data, pmavalues, probenames, genenames, dthresholdvalue,
				btakelog, bspotincluded, badd0, dsamplemins, sortedcorrvals,
				nmaxmissing, dmincorrelation, htFiltered, szProbeHeader,
				szGeneHeader, bmaxminval, otherInputFiles, bfullrepeat);
	}

	// ///////////////////////////////////////////////////////////////////////////////
	/**
	 *Given an array of otherDataSets, merges it with the current data set by
	 * storing in data the median of the values. If bfullrepeat is true then
	 * stores the repeat and missing data into generepeatspottimedata and
	 * generepeatspottimepma
	 */
	public DataSetCore mergeDataSets(DataSetCore[] otherDataSets) {
		otherInputFiles = new String[otherDataSets.length];
		for (int ndataset = 0; ndataset < otherDataSets.length; ndataset++) {
			otherInputFiles[ndataset] = otherDataSets[ndataset].szInputFile;
		}

		// assume that the otherDataSet has the same dimension as this dataset
		double[][] mergedata = new double[numrows][numcols];
		int[][] mergepmavalues = new int[numrows][numcols];
		double[] vals = new double[otherDataSets.length + 1];

		if (bfullrepeat) {
			// we've got distinct time series for which we need to store values
			generepeatspottimedata = new double[numrows][otherDataSets.length + 1][][];
			generepeatspottimepma = new int[numrows][otherDataSets.length + 1][][];
		} else {
			// the repeats are of the individual time points for which we will
			// take the mean? median
			generepeatspottimedata = null;
			generepeatspottimepma = null;
		}

		for (int nrow = 0; nrow < numrows; nrow++) {
			if (bfullrepeat) {
				generepeatspottimedata[nrow][0] = genespottimedata[nrow];
				generepeatspottimepma[nrow][0] = genespottimepma[nrow];
			}

			for (int ncol = 0; ncol < numcols; ncol++) {
				int nvalindex = 0;
				int npma = 0;
				if (pmavalues[nrow][ncol] != 0) {
					vals[nvalindex] = data[nrow][ncol];
					nvalindex++;
					npma = pmavalues[nrow][ncol];
				}

				for (int ndataset = 0; ndataset < otherDataSets.length; ndataset++) {
					if (bfullrepeat) {
						generepeatspottimedata[nrow][ndataset + 1] = otherDataSets[ndataset].genespottimedata[nrow];
						generepeatspottimepma[nrow][ndataset + 1] = otherDataSets[ndataset].genespottimepma[nrow];
					}
					if (otherDataSets[ndataset].pmavalues[nrow][ncol] != 0) {
						vals[nvalindex] = otherDataSets[ndataset].data[nrow][ncol];
						nvalindex++;
						// any non-zero pma makes it valid
						npma = Math.max(npma,
								otherDataSets[ndataset].pmavalues[nrow][ncol]);
					}
				}

				if (nvalindex > 0) {
					mergedata[nrow][ncol] = Util.getmedian(vals, nvalindex);
				}
				mergepmavalues[nrow][ncol] = npma;
			}
		}

		DataSetCore mergedSet = new DataSetCore(szInputFile, genespottimedata,
				generepeatspottimedata, genespottimepma, generepeatspottimepma,
				mergedata, mergepmavalues, probenames, genenames,
				dthresholdvalue, btakelog, bspotincluded, badd0, dsamplemins,
				sortedcorrvals, nmaxmissing, dmincorrelation, htFiltered,
				szProbeHeader, szGeneHeader, bmaxminval, otherInputFiles,
				bfullrepeat);

		return mergedSet;

	}

	// //////////////////////////////////////////////////////////////////////////////
	/**
	 *Computes the average pairwise correlation between gene repeats stores it
	 * in sortedcorrvals. Filters those genes which do not have a sortedcorrvals
	 * exceeding dmincorrelation
	 */
	public DataSetCore filterdistprofiles(DataSetCore theDataSet1,
			DataSetCore[] RepeatSet) {
		int npairs = (RepeatSet.length + 1) * (RepeatSet.length) / 2;
		double dweight = 1.0 / (double) npairs;
		int ngoodrows = 0;
		sortedcorrvals = new double[numrows];
		boolean[] goodrow = new boolean[numrows];
		for (int nrow = 0; nrow < sortedcorrvals.length; nrow++) {
			sortedcorrvals[nrow] = 0;
			for (int nrepeatset = 0; nrepeatset < RepeatSet.length; nrepeatset++) {
				sortedcorrvals[nrow] += dweight
						* Util.correlation(theDataSet1.data[nrow],
								RepeatSet[nrepeatset].data[nrow],
								theDataSet1.pmavalues[nrow],
								RepeatSet[nrepeatset].pmavalues[nrow]);
				for (int nrepeatset2 = nrepeatset + 1; nrepeatset2 < RepeatSet.length; nrepeatset2++) {
					sortedcorrvals[nrow] += dweight
							* Util.correlation(
									RepeatSet[nrepeatset].data[nrow],
									RepeatSet[nrepeatset2].data[nrow],
									RepeatSet[nrepeatset].pmavalues[nrow],
									RepeatSet[nrepeatset2].pmavalues[nrow]);
				}
			}

			if (sortedcorrvals[nrow] > dmincorrelation) {
				goodrow[nrow] = true;
				ngoodrows++;
			} else {
				goodrow[nrow] = false;
			}
		}

		Arrays.sort(sortedcorrvals);

		return filtergenesgeneral(goodrow, ngoodrows, true);
	}

	// //////////////////////////////////////////////////////////////////////////////
	/**
	 *Filters those rows which do not have a true in keepgene nkeep is the
	 * number of true rows in keepgene If bstore is true and gene is filtered
	 * then we stroe the gene and proble list for it in htFiltered Returns a new
	 * DataSetCore object with those rows filtered
	 */
	public DataSetCore filtergenesgeneral(boolean[] keepgene, int nkeep,
			boolean bstore) {
		if (nkeep < 1)
			throw new IllegalArgumentException("All Genes Filtered");

		double[][] filterdata = new double[nkeep][];
		int[][] filterpmavalues = new int[nkeep][];
		String[] filtergenenames = new String[nkeep];
		String[] filterprobenames = new String[nkeep];

		double[][][] filtergenespottimedata;
		double[][][][] filtergenerepeatspottimedata;
		int[][][] filtergenespottimepma;
		int[][][][] filtergenerepeatspottimepma;

		boolean bfiltergenespottime = ((genespottimedata != null) && (genespottimedata.length == numrows));
		boolean bfiltergenerepeatspottime = ((generepeatspottimedata != null) && (generepeatspottimedata.length == numrows));

		if (!bfiltergenespottime) {
			filtergenespottimedata = genespottimedata;
			filtergenespottimepma = genespottimepma;
		} else {
			filtergenespottimedata = new double[nkeep][][];
			filtergenespottimepma = new int[nkeep][][];
		}

		if (!bfiltergenerepeatspottime) {
			filtergenerepeatspottimedata = generepeatspottimedata;
			filtergenerepeatspottimepma = generepeatspottimepma;
		} else {
			filtergenerepeatspottimedata = new double[nkeep][][][];
			filtergenerepeatspottimepma = new int[nkeep][][][];
		}

		int nfilterindex = 0;
		for (int nrow = 0; nrow < numrows; nrow++) {
			if (keepgene[nrow]) {
				// this row is a keeper store over its info
				filtergenenames[nfilterindex] = genenames[nrow];
				filterprobenames[nfilterindex] = probenames[nrow];
				if (bfiltergenespottime) {
					filtergenespottimedata[nfilterindex] = genespottimedata[nrow];
					filtergenespottimepma[nfilterindex] = genespottimepma[nrow];
				}

				if (bfiltergenerepeatspottime) {
					filtergenerepeatspottimedata[nfilterindex] = generepeatspottimedata[nrow];
					filtergenerepeatspottimepma[nfilterindex] = generepeatspottimepma[nrow];
				}

				filterpmavalues[nfilterindex] = pmavalues[nrow];
				filterdata[nfilterindex] = data[nrow];

				nfilterindex++;
			} else if (bstore) {
				htFiltered.put(genenames[nrow], probenames[nrow]);
			}
		}

		return new DataSetCore(szInputFile, filtergenespottimedata,
				filtergenerepeatspottimedata, filtergenespottimepma,
				filtergenerepeatspottimepma, filterdata, filterpmavalues,
				filterprobenames, filtergenenames, dthresholdvalue, btakelog,
				bspotincluded, badd0, dsamplemins, sortedcorrvals, nmaxmissing,
				dmincorrelation, htFiltered, szProbeHeader, szGeneHeader,
				bmaxminval, otherInputFiles, bfullrepeat);
	}

	// ///////////////////////////////////////////////////////////////////////////////

	/***
	 *If bmaxminval is true, then filter those genes for which the difference
	 * between the max and min value is less than dthresholdvalue If bmaxminval
	 * if false, then filter those genes for which the absolute expression
	 * change is less than dmaxval
	 */
	public DataSetCore filtergenesthreshold2() {
		if (bmaxminval) {
			return filtergenesthreshold2maxmin();
		} else {
			return filtergenesthreshold2change();
		}
	}

	// //////////////////////////////////////////////////////////////////////////////

	/**
	 *Filter those genes for which the absolute expression change is less than
	 * dmaxval
	 */
	private DataSetCore filtergenesthreshold2change() {
		boolean[] expressedrows = new boolean[numrows];
		int nabovethreshold = 0;
		double dmaxval;

		for (int nrow = 0; nrow < numrows; nrow++) {
			dmaxval = 0;
			for (int ncol = 1; ncol < numcols; ncol++) {
				if ((pmavalues[nrow][ncol] > 0)
						&& (Math.abs(data[nrow][ncol]) > dmaxval)) {
					dmaxval = Math.abs(data[nrow][ncol]);
				}
			}

			if (dmaxval >= dthresholdvalue) {
				expressedrows[nrow] = true;
				nabovethreshold++;
			} else {
				expressedrows[nrow] = false;
			}
		}

		return filtergenesgeneral(expressedrows, nabovethreshold, true);
	}

	// /////////////////////////////////////////////////////////
	/**
	 *Filter those genes for which the difference between the max and min value
	 * is less than dthresholdvalue
	 */
	private DataSetCore filtergenesthreshold2maxmin() {
		boolean[] expressedrows = new boolean[numrows];
		int nabovethreshold = 0;

		for (int nrow = 0; nrow < numrows; nrow++) {
			double dmax = 0;
			double dmin = 0;
			for (int ncol = 1; ncol < numcols; ncol++) {
				if ((pmavalues[nrow][ncol] > 0) && (data[nrow][ncol] > dmax)) {
					dmax = data[nrow][ncol];
				}
				if ((pmavalues[nrow][ncol] > 0) && (data[nrow][ncol] < dmin)) {
					dmin = data[nrow][ncol];
				}
			}

			if (dmax - dmin >= dthresholdvalue) {
				expressedrows[nrow] = true;
				nabovethreshold++;
			} else {
				expressedrows[nrow] = false;
			}
		}

		return filtergenesgeneral(expressedrows, nabovethreshold, true);
	}

	// ///////////////////////////////////////////////////////////
	/**
	 *Filters those genes with expression below dthresholdvalue
	 */
	public DataSetCore filtergenesthreshold1point() {
		boolean[] expressedrows = new boolean[numrows];
		int nabovethreshold = 0;
		double dmaxval;

		for (int nrow = 0; nrow < numrows; nrow++) {
			dmaxval = Math.abs(data[nrow][0]);
			if (dmaxval >= dthresholdvalue) {
				expressedrows[nrow] = true;
				nabovethreshold++;
			} else {
				expressedrows[nrow] = false;
			}
		}

		return filtergenesgeneral(expressedrows, nabovethreshold, true);
	}
	// ///////////////////////////////////////////////////////////////////////////
}
