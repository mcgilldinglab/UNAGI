package edu.cmu.cs.sb.drem;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.Locale;
import java.util.StringTokenizer;
import java.util.zip.GZIPInputStream;

import edu.cmu.cs.sb.core.DataSetCore;
import java.io.FileWriter;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;


public class RegulatorBindingData {
	/**
	 * The three dimensional matrix giving the temporal interaction values for
	 * genes and regulators. First dimension: time point Second dimension: gene
	 * Third dimension: regulator
	 */
        public double[][][] gene2RegBinding;
	/**
	 * The signed version of gene2RegBinding. This gives the signed={-1,0,1}
	 * interaction value for the genes and regulators.
	 */
	public int[][][] gene2RegBindingSigned;
	/**
	 * The three dimensional matrix giving the temporal interaction values for
	 * genes and regulators. First dimension: time point Second dimension:
	 * regulator Third dimension: gene
	 */
	public double[][][] reg2GeneBinding;
	/**
	 * The signed version of reg2GeneBinding. This gives the signed={-1,0,1}
	 * interaction value for the genes and regulators.
	 */
	public int[][][] reg2GeneBindingSigned;
	/**
	 * The three dimensional matrix gives the indices for the gene that a
	 * regulator binds with. First dimension: time point Second dimension:
	 * regulator Third dimension: gene
	 */
	public int[][][] reg2GeneBindingIndex;
	/**
	 * The three dimensional matrix gives the indices for the regulator that a
	 * gene binds with. First dimension: time point Second dimension: gene Third
	 * dimension: regulator
	 */
	public int[][][] gene2RegBindingIndex;
	/**
	 * The union of the temporal binding data is taken. The binding value with
	 * the highest magnitude is stored.
	 */
	public double[][] gene2RegMaxBinding;
	/**
	 * The index for the union of temporal data.
	 */
	public int[][] gene2RegMaxBindingIndex;
	/**
	 * The union of the temporal regulator-wise binding data is taken. The
	 * binding value with the highest magnitude is stored.
	 */
	public double[][] reg2GeneMaxBinding;
	/**
	 * The index for the regulator-wise union of temporal data.
	 */
	public int[][] reg2GeneMaxBindingIndex;
	/**
	 * The number of Regulators aka the length of the regulatorNames array,
	 * previously numbits
	 */
	public int numberRegs;
	/**
	 * The number of genes in the data set.
	 */
	public int numberGenes;
	/**
	 * IDs of the regulators
	 */
	public String[] regNames;
	/**
	 * Synonyms of the names stored in regNames
	 */
	public HashMap<String, HashSet<String>> regSyns;
	public static final int TF = 0;
	public static final int MIRNA = 1;
	public static final String[] REGTYPE = { "TF", "MIRNA" };
	/**
	 * Denotes the types of the regulators in the regulatorNames array i.e.,
	 * 0=TF and 1=miRNA for each entry in RegulatorNames
	 */
	public int[] regTypes;
	/**
	 * Maps the regulator names to their regulator type.
	 */
	public HashMap<String, Integer> regTypeMap;
	/**
	 * Maps the regulator names to their row in the regulator binding data.
	 */
	public HashMap<String, Integer> regRowMap;
	/**
	 * Can be queried for existing regulator types in regulatorTypes Each
	 * regulator type is an Integer TF = 0 miRNA = 1
	 */
	public HashSet<Integer> existingRegTypes;
	/**
	 * Contains scores associated with regulators computed by a RegulatorScorer
	 */
	public double[] regPriors;
	/**
	 * Unsorted possible binding values previously hsUniqueInput
	 */
	public HashSet<Double> existingBindingValuesUnsorted;
	/**
	 * Sorted possible binding values previously dbindingvals
	 */
	public double[] existingBindingValuesSorted;
	/**
	 * Sorted possible signed binding values For use by the filtered classified
	 */
	public int[] signedBindingValuesSorted;
	/**
	 * Unsorted possible signed binding values For use by the filtered
	 * classified
	 */
	public HashSet<Integer> signedBindingValuesUnsorted;
	/**
	 * Each element is a geneID associated with a BindingGeneRec object
	 * Effectively a duplicate of gene2RegulatorBinding previously htBinding
	 */
	public HashMap<String, BindingGeneRec> geneID2RegBinding;
	/**
	 * The gene to regulator data set up in a format to be used by a weighted
	 * classifier. The first dimension is the number of children, the second is
	 * gene number, and the third is regulator number, with the value being the
	 * binding value.
	 */
	public double[][][][] theInstances;
	/**
	 * The gene to regulator index data set up in a format to be used by a
	 * weighted classifier. The first dimension is the number of children, the
	 * second is gene number, and the third is regulator number, with the value
	 * being the binding index
	 */
	public int[][][][] theInstancesIndex;
	/**
	 * The regulator to gene data set up in a format to be used by a weighted
	 * classifier. The first dimension is the number of children, the second is
	 * regulator number, and the third is gene number, with the value being the
	 * binding value.
	 */
	public double[][][][] theInstancesReg;
	/**
	 * The regulator to gene index data set up in a format to be used by a
	 * weighted classifier. The first dimension is the number of children, the
	 * second is regulator number, and the third is gene number, with the value
	 * being the binding index.
	 */
	public int[][][][] theInstancesRegIndex;
	public int[][][] ylabels;
	/**
	 * The number of timepoints in the expression data set
	 */
	public int timepoints;

	/*
	 * Fields to use for filtering the data set if bfilterbinding == true
	 */
	public int nbinding;
	public boolean[] bbindingdata;

	/**
	 * All TF activiy priors are adjusted by ACTIVITY_EPSILON to avoid divide by
	 * 0 errors when the TF activity prior is 1.
	 */
	static final double ACTIVITY_EPSILON = 1E-8;
	/**
	 * Default score for regulators not in the given file
	 */
	static final double DEFAULTREGSCORE = 0.5;

	private String SZDELIM;
	private boolean BDEBUG;
	private boolean BFILTERBINDING;

	/**
	 * When calling this constructor, the caller must filter the data set if
	 * bfilterbinding is set to true. The call is as follows: if(bfilterbinding)
	 * { theDataSet = new DREM_DataSet(theDataSet.filtergenesgeneral(
	 * bindingData.bbindingdata,bindingData.nbinding, true) , theDataSet.tga); }
	 * 
	 * @throws IOException
	 */
	public RegulatorBindingData(String tfBindingDataFile,
			String miRNABindingDataFile, boolean bfilterbinding,
			DREM_DataSet theds, String szdelim, boolean bdebug,
			String regScoreFile, String xrefFile,
			HashMap<String, Integer> timepointMap) throws IOException {

		SZDELIM = szdelim;
		geneID2RegBinding = new HashMap<String, BindingGeneRec>();
		existingBindingValuesUnsorted = new HashSet<Double>();

		BDEBUG = bdebug;
		BFILTERBINDING = bfilterbinding;

		numberRegs = 0;
		timepoints = theds.numcols;
		numberGenes = theds.numrows;
		regNames = new String[0];
		regTypes = new int[0];
		int offset = 0;

		String[] dataFiles = new String[2];
		dataFiles[TF] = tfBindingDataFile;
		dataFiles[MIRNA] = miRNABindingDataFile;

		for (int i = 0; i < dataFiles.length; i++) {
			if (dataFiles[i] != null && !dataFiles[i].equals("")) {
				BufferedReader br = null;
				try {
					br = new BufferedReader(new InputStreamReader(
							new GZIPInputStream(new FileInputStream(
									dataFiles[i]))));
				} catch (IOException ex) {
					br = new BufferedReader(new FileReader(dataFiles[i]));
				}

				String szLine = br.readLine();
				StringTokenizer st = new StringTokenizer(szLine, "\t");
				String szh1 = "";

				int numType;
				if (szLine == null) {
					throw new IllegalArgumentException(
							"Empty regulator-gene interaction input file found!");
				} else if (szLine.startsWith("\t")) {
					numType = st.countTokens();
				} else {
					numType = st.countTokens() - 1;
					szh1 = st.nextToken();
				}

				// Read the top row of the file in
				String[] tempRegNames = new String[numType];
				for (int nRegIndex = 0; st.hasMoreTokens(); nRegIndex++) {
					tempRegNames[nRegIndex] = st.nextToken();
				}

				/*
				 * If the top row of the file is 3 total words, the first of
				 * which is the regulator types, and the second of which is
				 * "GENE", then we are in the three column format.
				 */
				boolean bthreecol = (((numType == 2) || (numType == 3))
						&& (szh1.equalsIgnoreCase(REGTYPE[i])) && (tempRegNames[0]
						.equalsIgnoreCase("GENE")));
				boolean usingTimepoints = bthreecol && numType == 3;
				/*
				 * The parseThreeColFormat method expands the regulator names
				 * and regulator types array on its own, while the
				 * parseGridFormat, needs it done before hand.
				 */
				if (bthreecol) {
					parseThreeColFormat(br, offset, i, usingTimepoints,
							timepointMap);
				} else {
					numberRegs = offset + numType;
					String[] totalRegulatorNames = new String[offset + numType];
					for (int j = 0; j < offset; j++)
						totalRegulatorNames[j] = regNames[j];
					for (int j = offset; j < numberRegs; j++)
						totalRegulatorNames[j] = tempRegNames[j - offset];
					regNames = totalRegulatorNames;
					int[] totalRegulatorTypes = new int[offset + numType];
					for (int j = 0; j < offset; j++)
						totalRegulatorTypes[j] = regTypes[j];
					for (int j = offset; j < numberRegs; j++)
						totalRegulatorTypes[j] = i;
					regTypes = totalRegulatorTypes;
					numberRegs = regNames.length;
					parseGridFormat(br, offset, numType);
				}
				offset = numberRegs;
			}
		}

		existingBindingValuesUnsorted.add(new Double(0));
		/*
		 * The following code loads the recorded binding data into the
		 * gene2RegulatorBinding arrays, optionally filtering out genes from the
		 * DataSet object for which no binding data is available.
		 */
		if (BFILTERBINDING) {
			// filter those genes without binding data
			nbinding = 0; // count of the number of genes with binding info
			int count = 0;
			// first determining how many genes we have binding data for
			BindingGeneRec[] hitA = new BindingGeneRec[theds.data.length];
			for (int nrow = 0; nrow < theds.data.length; nrow++) {
				BindingGeneRec bgr = getBindingObject(theds.genenames[nrow],
						theds.probenames[nrow]);

				if (bgr != null) {
					hitA[nrow] = bgr;
					nbinding++;
				} else {
					hitA[nrow] = null;
					count++;
				}
			}

			System.out.println("filtering " + count + " genes");
			if (BDEBUG) {
				System.out.println("nbinding = " + nbinding);
			}

			gene2RegBinding = new double[timepoints][nbinding][];
			gene2RegBindingIndex = new int[timepoints][nbinding][];
			int nbindingindex = 0;
			System.out.println("nbinding: " + nbinding);
			// stores whether or not should keep gene
			bbindingdata = new boolean[theds.data.length];

			// going through all data values storing into
			// bindingpval binding values
			// flagging for filtering those records without binding
			for (int nrow = 0; nrow < theds.data.length; nrow++) {
				if (hitA[nrow] != null) {
					BindingGeneRec theBindingGeneRec = hitA[nrow];

					for (int time = 0; time < timepoints; time++) {
						gene2RegBinding[time][nbindingindex] = theBindingGeneRec.gene2RegulatorBindingRec[time];

						// System.out.println("Setting index " + nbindingindex +
						// " to "
						// +
						// theBindingGeneRec.gene2RegulatorBindingRecIndex[time]);
						if (theBindingGeneRec.gene2RegulatorBindingRecIndex[time] == null)
							System.out
									.println("ERROR BINDING REC INDEX IS NULL");

						gene2RegBindingIndex[time][nbindingindex] = theBindingGeneRec.gene2RegulatorBindingRecIndex[time];
						for (int nindex = 0; nindex < theBindingGeneRec.gene2RegulatorBindingRec[time].length; nindex++) {
							existingBindingValuesUnsorted
									.add(new Double(
											theBindingGeneRec.gene2RegulatorBindingRec[time][nindex]));
						}
					}
					nbindingindex++;
					bbindingdata[nrow] = true;
				} else {
					bbindingdata[nrow] = false;
				}
			}

			System.out.println("Final nbindingindex: " + nbindingindex);
			System.out.println();
		} else {
			// not filtering genes with missing p-values instead setting to 0
			// transfering binding p-values in a hashmap to an array
			// 0 values if do not have a p-value for that gene
			// bindingpval = new double[theDataSet.data.length][numbits];
			gene2RegBinding = new double[timepoints][theds.data.length][];
			gene2RegBindingIndex = new int[timepoints][theds.data.length][];

			for (int time = 0; time < timepoints; time++) {
				for (int nrow = 0; nrow < theds.data.length; nrow++) {
					Object obj = getBindingObject(theds.genenames[nrow],
							theds.probenames[nrow]);

					if (obj != null) {
						BindingGeneRec theBindingGeneRec = (BindingGeneRec) obj;
						gene2RegBinding[time][nrow] = theBindingGeneRec.gene2RegulatorBindingRec[time];
						gene2RegBindingIndex[time][nrow] = theBindingGeneRec.gene2RegulatorBindingRecIndex[time];
						for (int nindex = 0; nindex < theBindingGeneRec.gene2RegulatorBindingRec[time].length; nindex++) {
							existingBindingValuesUnsorted
									.add(new Double(
											theBindingGeneRec.gene2RegulatorBindingRec[time][nindex]));
						}
					} else {
						gene2RegBinding[time][nrow] = new double[0];
						gene2RegBindingIndex[time][nrow] = new int[0];
					}
				}
			}
		}

		// Initialize and create the regulator-wise arrays and signed arrays
		reg2GeneBindingIndex = new int[timepoints][numberRegs][];
		reg2GeneBinding = new double[timepoints][numberRegs][];
		gene2RegBindingSigned = new int[timepoints][][];
		reg2GeneBindingSigned = new int[timepoints][][];
		for (int time = 0; time < timepoints; time++) {
			makeRegulatorIndex(gene2RegBinding[time],
					gene2RegBindingIndex[time], reg2GeneBinding[time],
					reg2GeneBindingIndex[time]);
		}
		makeSigned(gene2RegBinding, gene2RegBindingSigned);
		makeSigned(reg2GeneBinding, reg2GeneBindingSigned);

		// Construct the sorted binding values array
		int nel = 0;
		existingBindingValuesSorted = new double[existingBindingValuesUnsorted
				.size()];
		for (Double d : existingBindingValuesUnsorted) {
			existingBindingValuesSorted[nel] = d.doubleValue();
			nel++;
		}
		Arrays.sort(existingBindingValuesSorted);

		// Contruct the sorted and unsorted signed binding value objects
		signedBindingValuesUnsorted = new HashSet<Integer>();
		for (Double d : existingBindingValuesUnsorted) {
			signedBindingValuesUnsorted.add(new Integer((int) Math.signum(d)));
		}
		signedBindingValuesSorted = new int[signedBindingValuesUnsorted.size()];
		int i = 0;
		for (Integer integer : signedBindingValuesUnsorted) {
			signedBindingValuesSorted[i] = integer.intValue();
			i++;
		}
		Arrays.sort(signedBindingValuesSorted);

		// Construct the regulator synonyms hash table
		HashSet<String> hsRegNames = new HashSet<String>();
		regSyns = new HashMap<String, HashSet<String>>();
		if (xrefFile != null && !xrefFile.equals("")) {
			for (int k = 0; k < regNames.length; k++) {
				hsRegNames.add(regNames[k].toUpperCase());
				regSyns.put(regNames[k].toUpperCase(), new HashSet<String>());
			}
			buildxref(hsRegNames, regSyns, xrefFile);
		}

		// Create a map from regulator name to binding data row
		regRowMap = new HashMap<String, Integer>();
		for (int j = 0; j < regNames.length; j++)
			regRowMap.put(regNames[j], j);

		// If we have a regulator score file create the score array
		if (regScoreFile != null && !regScoreFile.equals("")) {
			regPriors = new double[numberRegs];
			boolean[] scored = new boolean[numberRegs];
			double maxPrior = 0;
			BufferedReader br = null;
			try {
				br = new BufferedReader(new InputStreamReader(
						new GZIPInputStream(new FileInputStream(regScoreFile))));
			} catch (IOException ex) {
				br = new BufferedReader(new FileReader(regScoreFile));
			}
			String line;
			while ((line = br.readLine()) != null) {
				StringTokenizer st = new StringTokenizer(line, "\t");
				if (st.countTokens() != 2)
					throw new IOException("The regulator score file should"
							+ " have two columns: <regulator>\t<score>");
				String reg = st.nextToken();
				double score = Double.parseDouble(st.nextToken());
				Integer row = regRowMap.get(reg);
				if (row == null) {
					if (regSyns.containsKey(reg)) {
						for (String syn : regSyns.get(reg)) {
							row = regRowMap.get(syn);
							if (row != null)
								break;
						}
					}
					if (row == null) {
						throw new IOException(
								"All regulators in the regulator score "
										+ "should either be in or have a synonym in the regulator"
										+ " binding data file.");
					}
				}
				regPriors[row] = score;
				scored[row] = true;
				maxPrior = Math.max(maxPrior, score);
			}
			// Activity priors must be in the range [0,1]. Because the absolute
			// value of the binding values was used, no prior will be < 0.
			// However, if any value is > 1, the priors must be normalized.
			if (maxPrior > 1) {
				for (int j = 0; j < regPriors.length; j++)
					regPriors[j] /= maxPrior;
			}
			// To avoid dividing by 0 when a TF activity prior is 1, adjust
			// all priors by ACTIVITY_EPSILON. Priors are allowed to be 0.
			for (int ntf = 0; ntf < regPriors.length; ntf++) {
				if (scored[ntf]) {
					if (regPriors[ntf] > ACTIVITY_EPSILON) {
						regPriors[ntf] -= ACTIVITY_EPSILON;
					} else {
						regPriors[ntf] = 0;
					}
				} else {
					regPriors[ntf] = DEFAULTREGSCORE;
				}
			}
		}

		// Construct the set of regulator types
		existingRegTypes = new HashSet<Integer>();
		for (int j = 0; j < regTypes.length; j++) {
			existingRegTypes.add(regTypes[j]);
		}

		// Construct the regulator to type map
		regTypeMap = new HashMap<String, Integer>();
		for (int j = 0; j < regTypes.length; j++) {
			regTypeMap.put(regNames[j], regTypes[j]);
		}

		// Construct the max,union of the temporal data
		gene2RegMaxBinding = new double[gene2RegBinding[0].length][];
		gene2RegMaxBindingIndex = new int[gene2RegBinding[0].length][];
		for (int g = 0; g < gene2RegBinding[0].length; g++) {
			double[] maxes = new double[numberRegs];
			int countnonzero = 0;
			for (int t = 0; t < gene2RegBinding.length; t++) {
				for (int r = 0; r < gene2RegBinding[t][g].length; r++) {
					if (gene2RegBinding[t][g][r] != 0) {
						int index = gene2RegBindingIndex[t][g][r];
						if (maxes[index] == 0) {
							countnonzero++;
							maxes[index] = gene2RegBinding[t][g][r];
						} else if (Math.abs(gene2RegBinding[t][g][r]) > Math
								.abs(maxes[index])) {
							maxes[index] = gene2RegBinding[t][g][r];
						}
					}
				}
			}

			gene2RegMaxBinding[g] = new double[countnonzero];
			gene2RegMaxBindingIndex[g] = new int[countnonzero];

			int count = 0;
			for (int r = 0; r < numberRegs; r++) {
				if (maxes[r] != 0) {
					gene2RegMaxBinding[g][count] = maxes[r];
					gene2RegMaxBindingIndex[g][count] = r;
					count++;
				}
			}
		}
		reg2GeneMaxBinding = new double[numberRegs][];
		reg2GeneMaxBindingIndex = new int[numberRegs][];
		makeRegulatorIndex(gene2RegMaxBinding, gene2RegMaxBindingIndex,
				reg2GeneMaxBinding, reg2GeneMaxBindingIndex);
	}

	/**
	 * This constructor build a binding data out of another one based on a
	 * boolean array, keeping the true entries. This is used to create the train
	 * and test data.
	 * 
	 * @param total
	 *            The original RegulatorBindingData to build from.
	 * @param keep
	 *            The boolean array with true entries meaning keep this index.
	 * @param keepCount
	 *            The number of total entries to be kept.
	 */
	public RegulatorBindingData(RegulatorBindingData total, boolean[] keep,
			int keepCount, boolean removeGenes) {
		if (removeGenes) {
			// Copy in the binding data that we want to keep
			timepoints = total.timepoints;
			gene2RegBinding = new double[timepoints][keepCount][];
			gene2RegBindingIndex = new int[timepoints][keepCount][];
			existingBindingValuesUnsorted = new HashSet<Double>();
			for (int time = 0; time < timepoints; time++) {
				int keepIndex = 0;
				for (int i = 0; i < keep.length; i++) {
					if (keep[i]) {
						gene2RegBinding[time][keepIndex] = new double[total.gene2RegBinding[time][i].length];
						gene2RegBindingIndex[time][keepIndex] = new int[total.gene2RegBindingIndex[time][i].length];
						for (int j = 0; j < total.gene2RegBinding[time][i].length; j++) {
							gene2RegBinding[time][keepIndex][j] = total.gene2RegBinding[time][i][j];
							gene2RegBindingIndex[time][keepIndex][j] = total.gene2RegBindingIndex[time][i][j];
							existingBindingValuesUnsorted
									.add(gene2RegBinding[time][keepIndex][j]);
						}
						keepIndex++;
					}
				}
			}
			// Construct the remaining fields
			numberRegs = total.numberRegs;
			reg2GeneBinding = new double[timepoints][numberRegs][];
			reg2GeneBindingIndex = new int[timepoints][numberRegs][];
			gene2RegBindingSigned = new int[timepoints][][];
			reg2GeneBindingSigned = new int[timepoints][][];
			for (int time = 0; time < timepoints; time++) {
				makeRegulatorIndex(gene2RegBinding[time],
						gene2RegBindingIndex[time], reg2GeneBinding[time],
						reg2GeneBindingIndex[time]);
			}
			makeSigned(gene2RegBinding, gene2RegBindingSigned);
			makeSigned(reg2GeneBinding, reg2GeneBindingSigned);

			regNames = new String[numberRegs];
			regTypes = new int[numberRegs];
			if (total.regPriors != null) {
				regPriors = new double[numberRegs];
				for (int i = 0; i < numberRegs; i++)
					regPriors[i] = total.regPriors[i];
			}
			existingRegTypes = new HashSet<Integer>();
			regRowMap = new HashMap<String, Integer>();
			for (int i = 0; i < numberRegs; i++) {
				regNames[i] = total.regNames[i];
				regTypes[i] = total.regTypes[i];
				existingRegTypes.add(regTypes[i]);
				regRowMap.put(regNames[i], i);
			}
			existingBindingValuesSorted = new double[existingBindingValuesUnsorted
					.size()];
			int k = 0;
			for (Double d : existingBindingValuesUnsorted) {
				existingBindingValuesSorted[k] = d;
				k++;
			}
			Arrays.sort(existingBindingValuesSorted);

			signedBindingValuesUnsorted = new HashSet<Integer>();
			for (Double d : existingBindingValuesUnsorted) {
				signedBindingValuesUnsorted.add((int) Math.signum(d));
			}

			signedBindingValuesSorted = new int[signedBindingValuesUnsorted
					.size()];
			int i = 0;
			for (Integer integer : signedBindingValuesUnsorted) {
				signedBindingValuesSorted[i] = integer.intValue();
				i++;
			}
			Arrays.sort(signedBindingValuesSorted);

			regSyns = total.regSyns;
			regTypeMap = total.regTypeMap;

			// Construct the max,union of the temporal data
			numberGenes = keepCount;
			gene2RegMaxBinding = new double[keepCount][];
			gene2RegMaxBindingIndex = new int[keepCount][];
			for (int g = 0; g < keepCount; g++) {
				double[] maxes = new double[numberRegs];
				int countnonzero = 0;
				for (int t = 0; t < timepoints; t++) {
					for (int r = 0; r < gene2RegBinding[t][g].length; r++) {
						if (gene2RegBinding[t][g][r] != 0) {
							int index = gene2RegBindingIndex[t][g][r];
							if (maxes[index] == 0) {
								countnonzero++;
								maxes[index] = gene2RegBinding[t][g][r];
							} else if (Math.abs(gene2RegBinding[t][g][r]) > Math
									.abs(maxes[index])) {
								maxes[index] = gene2RegBinding[t][g][r];
							}
						}
					}
				}
				gene2RegMaxBinding[g] = new double[countnonzero];
				gene2RegMaxBindingIndex[g] = new int[countnonzero];

				int count = 0;
				for (int r = 0; r < numberRegs; r++) {
					if (maxes[r] != 0) {
						gene2RegMaxBinding[g][count] = maxes[r];
						gene2RegMaxBindingIndex[g][count] = r;
						count++;
					}
				}
			}
		} else { // Remove regulators
			System.out.println("FILTERING MIRNA");
			for (int i = 0; i < keep.length; i++) {
				System.out.println("Reg: " + total.regNames[i]);
				System.out.println("\tKeeping? " + keep[i]);
			}

			numberGenes = total.numberGenes;
			timepoints = total.timepoints;
			existingBindingValuesUnsorted = new HashSet<Double>();
			int[] newRegIndex = new int[keep.length];
			int newIndex = 0;
			for (int i = 0; i < newRegIndex.length; i++) {
				if (keep[i]) {
					newRegIndex[i] = newIndex;
					newIndex++;

					System.out.println("New index for " + i + " is "
							+ newRegIndex[i]);
				} else {
					newRegIndex[i] = -1;
				}
			}
			numberRegs = newIndex;
			gene2RegBinding = new double[timepoints][numberGenes][];
			gene2RegBindingIndex = new int[timepoints][numberGenes][];
			for (int i = 0; i < total.gene2RegBinding.length; i++) {
				for (int j = 0; j < total.gene2RegBinding[i].length; j++) {
					int count = 0;
					for (int k = 0; k < total.gene2RegBinding[i][j].length; k++) {
						if (keep[total.gene2RegBindingIndex[i][j][k]]) {
							count++;
						}
					}
					gene2RegBinding[i][j] = new double[count];
					gene2RegBindingIndex[i][j] = new int[count];
					count = 0;
					for (int k = 0; k < total.gene2RegBinding[i][j].length; k++) {
						if (keep[total.gene2RegBindingIndex[i][j][k]]) {
							gene2RegBinding[i][j][count] = total.gene2RegBinding[i][j][k];
							gene2RegBindingIndex[i][j][count] = newRegIndex[total.gene2RegBindingIndex[i][j][k]];
							existingBindingValuesUnsorted
									.add(total.gene2RegBinding[i][j][k]);
							count++;
						}
					}
				}
			}

			for (int i = 0; i < gene2RegBinding[0].length; i++) {
				System.out.println("Gene " + i);
				for (int j = 0; j < gene2RegBinding[0][i].length; j++) {
					System.out
							.println("\tReg " + gene2RegBindingIndex[0][i][j]);
				}
			}
			// Construct the remaining fields
			reg2GeneBinding = new double[timepoints][numberRegs][];
			reg2GeneBindingIndex = new int[timepoints][numberRegs][];
			gene2RegBindingSigned = new int[timepoints][][];
			reg2GeneBindingSigned = new int[timepoints][][];
			for (int time = 0; time < timepoints; time++) {
				makeRegulatorIndex(gene2RegBinding[time],
						gene2RegBindingIndex[time], reg2GeneBinding[time],
						reg2GeneBindingIndex[time]);
			}
			makeSigned(gene2RegBinding, gene2RegBindingSigned);
			makeSigned(reg2GeneBinding, reg2GeneBindingSigned);

			regNames = new String[numberRegs];
			regTypes = new int[numberRegs];
			if (total.regPriors != null) {
				regPriors = new double[numberRegs];
				for (int i = 0; i < total.numberRegs; i++) {
					if (keep[i])
						regPriors[newRegIndex[i]] = total.regPriors[i];
				}
			}
			existingRegTypes = new HashSet<Integer>();
			regRowMap = new HashMap<String, Integer>();
			for (int i = 0; i < total.numberRegs; i++) {
				if (keep[i]) {
					regNames[newRegIndex[i]] = total.regNames[i];
					regTypes[newRegIndex[i]] = total.regTypes[i];
					existingRegTypes.add(regTypes[newRegIndex[i]]);
					regRowMap.put(regNames[newRegIndex[i]], newRegIndex[i]);
				}
			}
			existingBindingValuesSorted = new double[existingBindingValuesUnsorted
					.size()];
			int k = 0;
			for (Double d : existingBindingValuesUnsorted) {
				existingBindingValuesSorted[k] = d;
				k++;
			}
			Arrays.sort(existingBindingValuesSorted);

			signedBindingValuesUnsorted = new HashSet<Integer>();
			for (Double d : existingBindingValuesUnsorted) {
				signedBindingValuesUnsorted.add((int) Math.signum(d));
			}

			signedBindingValuesSorted = new int[signedBindingValuesUnsorted
					.size()];
			int i = 0;
			for (Integer integer : signedBindingValuesUnsorted) {
				signedBindingValuesSorted[i] = integer.intValue();
				i++;
			}
			Arrays.sort(signedBindingValuesSorted);

			regSyns = total.regSyns;
			regTypeMap = total.regTypeMap;

			// Construct the max,union of the temporal data
			numberGenes = total.numberGenes;
			gene2RegMaxBinding = new double[numberGenes][];
			gene2RegMaxBindingIndex = new int[numberGenes][];
			for (int g = 0; g < numberGenes; g++) {
				double[] maxes = new double[numberRegs];
				int countnonzero = 0;
				for (int t = 0; t < timepoints; t++) {
					for (int r = 0; r < gene2RegBinding[t][g].length; r++) {
						if (gene2RegBinding[t][g][r] != 0) {
							int index = gene2RegBindingIndex[t][g][r];
							if (maxes[index] == 0) {
								countnonzero++;
								maxes[index] = gene2RegBinding[t][g][r];
							} else if (Math.abs(gene2RegBinding[t][g][r]) > Math
									.abs(maxes[index])) {
								maxes[index] = gene2RegBinding[t][g][r];
							}
						}
					}
				}
				gene2RegMaxBinding[g] = new double[countnonzero];
				gene2RegMaxBindingIndex[g] = new int[countnonzero];

				int count = 0;
				for (int r = 0; r < numberRegs; r++) {
					if (maxes[r] != 0) {
						gene2RegMaxBinding[g][count] = maxes[r];
						gene2RegMaxBindingIndex[g][count] = r;
						count++;
					}
				}
			}
			System.out.println("FILTERING FINISHED");
			System.out.println("Number regs: " + numberRegs);
			for (i = 0; i < regNames.length; i++) {
				System.out.println("Reg: " + regNames[i]);
				System.out.println("Binding for reg " + i + ":");
				for (int j = 0; j < reg2GeneBinding[0][i].length; j++) {
					System.out.println("\tBinds "
							+ reg2GeneBindingIndex[0][i][j]);
				}
			}
		}
	}

	/**
	 * Record of index and regulatory value
	 */
	private static class RegulatorRec {
		int nRegulatorIndex;
		double nRegVal;

		RegulatorRec(int nRegulatorIndex, double nRegVal) {
			this.nRegulatorIndex = nRegulatorIndex;
			this.nRegVal = nRegVal;
		}
	}

	/**
	 * Comparator object sorts strictly based on nRegulatorIndex
	 */
	private class RegulatorRecCompare implements Comparator<RegulatorRec> {
		public int compare(RegulatorRec regr1, RegulatorRec regr2) {
			if (regr1.nRegulatorIndex < regr2.nRegulatorIndex) {
				return -1;
			} else if (regr1.nRegulatorIndex > regr2.nRegulatorIndex) {
				return 1;
			} else {
				return 0;
			}
		}
	}

	/**
	 * Reads in the regulator-gene interaction data currently in the format
	 * regulator, gene, and if specified interaction value where each column is
	 * delimitted by tabs
	 */
	private void parseThreeColFormat(BufferedReader br, int offset, int type,
			boolean usingTimepoints, HashMap<String, Integer> timepointMap)
			throws IOException {
		HashMap<GeneAndTimepoint, ArrayList<RegulatorRec>> htGeneTimeToRegulatorArray = new HashMap<GeneAndTimepoint, ArrayList<RegulatorRec>>();
		HashMap<String, Integer> htRegulatorToInteger = new HashMap<String, Integer>();
		// Start at offset so mirna data will have differenet indicies from tfs
		int nReg = offset;
		String szLine;
		while ((szLine = br.readLine()) != null) {
			StringTokenizer st = new StringTokenizer(szLine, "\t");
			String szRegulator = st.nextToken();
			String szGene = st.nextToken();
			double ninput;
			if (st.hasMoreTokens()) {
				String szToken = st.nextToken();
				try {
					ninput = Double.parseDouble(szToken);
				} catch (NumberFormatException nfex) {
					throw new IllegalArgumentException(szToken + " is not a"
							+ " valid score for a regulator-gene interaction");
				}
			} else {
				ninput = 1.0;
			}

			Integer timepoint = -1;
			if (usingTimepoints) {
				String time = "";
				if (st.hasMoreTokens()) {
					time = st.nextToken();
				} else {
					throw new IOException(
							"Timepoints should be associated with every entry.");
				}
				timepoint = timepointMap.get(time.toUpperCase());
				if (timepoint == null && time == "0")
					timepoint = 0;
				if (timepoint == null)
					throw new IOException(
							"Please annotate all binding data with a timepoint from"
									+ " the expression data.  " + timepoint
									+ " is not a valid timepoint.");
			}

			Integer objInt = htRegulatorToInteger.get(szRegulator);
			int nCurrRegulator;
			if (objInt == null) {
				nCurrRegulator = nReg;
				htRegulatorToInteger.put(szRegulator, new Integer(nReg));
				nReg++;
			} else {
				nCurrRegulator = ((Integer) objInt).intValue();
			}
			GeneAndTimepoint gat = new GeneAndTimepoint(szGene, timepoint);
			ArrayList<RegulatorRec> al = htGeneTimeToRegulatorArray.get(gat);
			if (al == null) {
				al = new ArrayList<RegulatorRec>();
			}
			al.add(new RegulatorRec(nCurrRegulator, ninput));
			htGeneTimeToRegulatorArray.put(gat, al);
		}
		br.close();
		if (offset > 0) {
			// If we are offset, then we need to create new larger arrays
			int numberNewRegulators = htRegulatorToInteger.size();
			String[] totalRegulatorNames = new String[offset
					+ numberNewRegulators];
			for (int i = 0; i < offset; i++)
				totalRegulatorNames[i] = regNames[i];
			regNames = totalRegulatorNames;
			int[] totalRegulatorTypes = new int[offset + numberNewRegulators];
			for (int i = 0; i < offset; i++)
				totalRegulatorTypes[i] = regTypes[i];
			regTypes = totalRegulatorTypes;
			numberRegs = regNames.length;
		} else {
			numberRegs = htRegulatorToInteger.size();
			regNames = new String[numberRegs];
			regTypes = new int[numberRegs];
		}
		Iterator<String> itrRegulatorSet = htRegulatorToInteger.keySet()
				.iterator();

		while (itrRegulatorSet.hasNext()) {
			String szRegulator = itrRegulatorSet.next();
			int index = ((Integer) htRegulatorToInteger.get(szRegulator))
					.intValue();
			regNames[index] = szRegulator;
			regTypes[index] = type;
		}

		Iterator<GeneAndTimepoint> itrGeneSet = htGeneTimeToRegulatorArray
				.keySet().iterator();
		while (itrGeneSet.hasNext()) {
			GeneAndTimepoint gat = itrGeneSet.next();
			ArrayList<RegulatorRec> al = htGeneTimeToRegulatorArray.get(gat);
			int numnonzero = al.size();

			RegulatorRec[] thePairsRecs = new RegulatorRec[numnonzero];
			for (int nbit = 0; nbit < numnonzero; nbit++) {
				thePairsRecs[nbit] = (RegulatorRec) al.get(nbit);
			}
			Arrays.sort(thePairsRecs, new RegulatorRecCompare());
			int[] nonzeroindex = new int[numnonzero];
			double[] nonzerovals = new double[numnonzero];
			for (int i = 0; i < numnonzero; i++) {
				RegulatorRec theRegulatorRec = thePairsRecs[i];
				nonzeroindex[i] = theRegulatorRec.nRegulatorIndex;
				nonzerovals[i] = theRegulatorRec.nRegVal;
			}
			loadBinding(gat.gene, nonzeroindex, nonzerovals, numnonzero,
					gat.timepoint);
		}
	}

	class GeneAndTimepoint {
		String gene;
		int timepoint;

		public GeneAndTimepoint(String gene, int timepoint) {
			this.gene = gene;
			this.timepoint = timepoint;
		}

		public int hashCode() {
			return gene.hashCode() + timepoint * 23;
		}
	}

	/**
	 * Reads in the regulator-gene input data assuming the header is already
	 * read and the data is in grid format
	 */
	private void parseGridFormat(BufferedReader br, int offset, int regCount)
			throws IOException {
		int[] nonzeroindex = new int[regCount];
		double[] nonzerovals = new double[regCount];
		String szLine;
		StringTokenizer st;

		while ((szLine = br.readLine()) != null) {
			int numnonzero = 0;

			if (!szLine.trim().equals("")) {
				st = new StringTokenizer(szLine, "\t");
				int numTokens = st.countTokens();
				if (regCount != (numTokens - 1)) {
					throw new IllegalArgumentException("Found a line with "
							+ (numTokens - 1) + " entries, expecting "
							+ numberRegs);
				}
				String szid = st.nextToken();

				for (int nindex = 0; nindex < regCount; nindex++) {
					String sznum = st.nextToken();
					try {
						double tempval = Double.parseDouble(sznum);
						if (tempval != 0) {
							nonzeroindex[numnonzero] = offset + nindex;
							nonzerovals[numnonzero] = tempval;
							numnonzero++;
						}
					} catch (NumberFormatException ex) {
						if ((numberRegs == 2) && (nindex == 0)) {
							throw new IllegalArgumentException(
									"If regulator-gene data is in column format,"
											+ " then the first two columns must have the "
											+ "headers 'Regulator Type' and 'Gene'");
						} else {
							throw new IllegalArgumentException(sznum
									+ " is not a valid value for a "
									+ "regulator-gene interaction!");
						}
					}
				}
				// storing each vector of binding values in hashtable with gene
				// identifier
				loadBinding(szid, nonzeroindex, nonzerovals, numnonzero, -1);
			}
		}
		br.close();
	}

	/**
	 * A recording contain an array of interaction values and another array
	 * containing the indices they correspond to
	 */
	static class BindingGeneRec {
		double[][] gene2RegulatorBindingRec;
		int[][] gene2RegulatorBindingRecIndex;
	}

	/**
	 * Stores into geneID2RegulatorBinding the mapping of the gene szid to its
	 * non-zero regulator-gene interactions given by nonzeroindex and
	 * nonzerovals
	 */
	public void loadBinding(String szid, int[] nonzeroindex,
			double[] nonzerovals, int numnonzero, int timepoint) {
		String szfull = szid.toUpperCase(Locale.ENGLISH);
		StringTokenizer stIDs = new StringTokenizer(szfull, SZDELIM);

		BindingGeneRec rec = new BindingGeneRec();
		rec.gene2RegulatorBindingRecIndex = new int[timepoints][numnonzero];
		rec.gene2RegulatorBindingRec = new double[timepoints][numnonzero];
		for (int i = 0; i < timepoints; i++) {
			if (timepoint == -1 || i == timepoint) {
				for (int j = 0; j < numnonzero; j++) {
					rec.gene2RegulatorBindingRecIndex[i][j] = nonzeroindex[j];
					rec.gene2RegulatorBindingRec[i][j] = nonzerovals[j];
				}

			} else {
				rec.gene2RegulatorBindingRec[i] = new double[0];
				rec.gene2RegulatorBindingRecIndex[i] = new int[0];
			}
		}

		while (stIDs.hasMoreTokens()) {
			// union if gene matches multiple hits in binding file, takes last
			// one seen if multiple non-zero
			String sztoken = stIDs.nextToken();
			BindingGeneRec currec = geneID2RegBinding.get(sztoken);

			if (currec != null) {
				rec = mergeArrays(rec.gene2RegulatorBindingRec,
						rec.gene2RegulatorBindingRecIndex,
						currec.gene2RegulatorBindingRec,
						currec.gene2RegulatorBindingRecIndex);
			}
			if (rec.gene2RegulatorBindingRec == null)
				System.out.println("ERROR binding array is null");
			if (rec.gene2RegulatorBindingRecIndex == null)
				System.out.println("ERROR index array is null");
			for (int i = 0; i < rec.gene2RegulatorBindingRec.length; i++) {
				if (rec.gene2RegulatorBindingRec[i] == null)
					System.out.println("ERROR binding " + i + " array is null");
				if (rec.gene2RegulatorBindingRecIndex[i] == null)
					System.out.println("ERROR index " + i + " array is null");
			}
			geneID2RegBinding.put(sztoken, rec);

			StringTokenizer stu = new StringTokenizer(sztoken, "_");
			if (stu.countTokens() > 1) {
				String szfirsttoken = stu.nextToken();
				currec = geneID2RegBinding.get(szfirsttoken);
				if (currec != null) {
					rec = mergeArrays(rec.gene2RegulatorBindingRec,
							rec.gene2RegulatorBindingRecIndex,
							currec.gene2RegulatorBindingRec,
							currec.gene2RegulatorBindingRecIndex);
				}
				if (rec.gene2RegulatorBindingRec == null)
					System.out.println("ERROR array is null");
				if (rec.gene2RegulatorBindingRecIndex == null)
					System.out.println("ERROR index array is null");
				for (int i = 0; i < rec.gene2RegulatorBindingRec.length; i++) {
					if (rec.gene2RegulatorBindingRec[i] == null)
						System.out.println("ERROR binding " + i
								+ " array is null");
					if (rec.gene2RegulatorBindingRecIndex[i] == null)
						System.out.println("ERROR index " + i
								+ " array is null");
				}
				geneID2RegBinding.put(szfirsttoken, rec);
			}
		}
	}

	/**
	 * Returns an entry of geneID2RegulatorBinding corresponding to szprobename
	 * or szgenename
	 */
	public BindingGeneRec getBindingObject(String szgenename, String szprobename) {

		StringTokenizer st = new StringTokenizer(szprobename, SZDELIM);
		String sztoken;
		BindingGeneRec bgr = null;

		while ((st.hasMoreTokens()) && (bgr == null)) {
			sztoken = st.nextToken();
			bgr = geneID2RegBinding.get(sztoken);
		}

		if (bgr == null) {
			st = new StringTokenizer(szgenename, SZDELIM);
			while ((st.hasMoreTokens()) && (bgr == null)) {
				sztoken = st.nextToken();
				bgr = geneID2RegBinding.get(sztoken);
			}
		}

		return bgr;
	}

	/**
	 * Assume geneindex1 and geneindex2 are sorted arrays Returns a
	 * BindingGeneRec where bindingpvalGeneindex has the union of geneindex1 and
	 * geneindex2 and bindingpvalGene has the corresponding values using the
	 * maximum to resolve disagreements
	 */
	private BindingGeneRec mergeArrays(double[][] geneval1, int[][] geneindex1,
			double[][] geneval2, int[][] geneindex2) {
		BindingGeneRec theBindingGeneRec = new BindingGeneRec();
		theBindingGeneRec.gene2RegulatorBindingRec = new double[timepoints][];
		theBindingGeneRec.gene2RegulatorBindingRecIndex = new int[timepoints][];

		for (int i = 0; i < timepoints; i++) {
			int nindex1 = 0;
			int nindex2 = 0;
			int nmatch = 0;
			while ((nindex1 < geneindex1[i].length)
					&& (nindex2 < geneindex2[i].length)) {
				if (geneindex1[i][nindex1] == geneindex2[i][nindex2]) {
					nmatch++;
					nindex1++;
					nindex2++;
				} else if (geneindex1[i][nindex1] < geneindex2[i][nindex2]) {
					nindex1++;
				} else if (geneindex2[i][nindex2] < geneindex1[i][nindex1]) {
					nindex2++;
				}
			}
			int nmergecount = geneindex1[i].length + geneindex2[i].length
					- nmatch;
			theBindingGeneRec.gene2RegulatorBindingRec[i] = new double[nmergecount];
			theBindingGeneRec.gene2RegulatorBindingRecIndex[i] = new int[nmergecount];
			int nmergeindex = 0;
			nindex1 = 0;
			nindex2 = 0;
			while ((nindex1 < geneindex1[i].length)
					&& (nindex2 < geneindex2[i].length)) {
				if (geneindex1[i][nindex1] == geneindex2[i][nindex2]) {
					theBindingGeneRec.gene2RegulatorBindingRec[i][nmergeindex] = Math
							.max(geneval1[i][nindex1], geneval2[i][nindex2]);
					theBindingGeneRec.gene2RegulatorBindingRecIndex[i][nmergeindex] = geneindex2[i][nindex2];
					nindex1++;
					nindex2++;
				} else if (geneindex1[i][nindex1] < geneindex2[i][nindex2]) {
					theBindingGeneRec.gene2RegulatorBindingRec[i][nmergeindex] = geneval1[i][nindex1];
					theBindingGeneRec.gene2RegulatorBindingRecIndex[i][nmergeindex] = geneindex1[i][nindex1];
					nindex1++;
				} else if (geneindex2[i][nindex2] < geneindex1[i][nindex1]) {
					theBindingGeneRec.gene2RegulatorBindingRec[i][nmergeindex] = geneval2[i][nindex2];
					theBindingGeneRec.gene2RegulatorBindingRecIndex[i][nmergeindex] = geneindex2[i][nindex2];
					nindex2++;
				}
				nmergeindex++;
			}

			while (nindex2 < geneindex2[i].length) {
				theBindingGeneRec.gene2RegulatorBindingRec[i][nmergeindex] = geneval2[i][nindex2];
				theBindingGeneRec.gene2RegulatorBindingRecIndex[i][nmergeindex] = geneindex2[i][nindex2];
				nmergeindex++;
				nindex2++;
			}
			while (nindex1 < geneindex1[i].length) {
				theBindingGeneRec.gene2RegulatorBindingRec[i][nmergeindex] = geneval1[i][nindex1];
				theBindingGeneRec.gene2RegulatorBindingRecIndex[i][nmergeindex] = geneindex1[i][nindex1];
				nmergeindex++;
				nindex1++;
			}
		}
		return theBindingGeneRec;
	}

	/**
	 * Generates valReg and valRegIndex based on val and valIndex so that the
	 * rows correspond to each regulator and each entry in a row valRegIndex
	 * corresponds to an index of a gene the regulator regulates and the
	 * corresponding entry in valReg corresponds to the interaction value
	 */
	public void makeRegulatorIndex(double[][] val, int[][] valIndex,
			double[][] valReg, int[][] valRegIndex) {
		int[] counts = new int[valReg.length];
		int sum = 0;
		for (int i = 0; i < valIndex.length; i++) {
			if (valIndex[i] == null)
				System.out.println("Null index is: " + i);
			for (int j = 0; j < valIndex[i].length; j++) {
				counts[valIndex[i][j]]++;
				sum++;
			}
		}

		for (int i = 0; i < valRegIndex.length; i++) {
			valReg[i] = new double[counts[i]];
			valRegIndex[i] = new int[counts[i]];
		}

		// reusing counts as number found so far
		for (int i = 0; i < counts.length; i++) {
			counts[i] = 0;
		}

		for (int ngene = 0; ngene < valIndex.length; ngene++) {
			for (int nregindex = 0; nregindex < valIndex[ngene].length; nregindex++) {
				int nReg = valIndex[ngene][nregindex];
				int nRegindex = counts[valIndex[ngene][nregindex]];
				valReg[nReg][nRegindex] = val[ngene][nregindex];
				valRegIndex[nReg][nRegindex] = ngene;
				counts[nReg]++;
			}
		}
	}

	/**
	 * Takes in a values array and fills in the Signed array with the signs of
	 * the entries in values {-1,0,1}
	 */
	public void makeSigned(double[][][] values, int[][][] valuesSigned) {
		for (int t = 0; t < timepoints; t++) {
			valuesSigned[t] = new int[values[t].length][];
			for (int i = 0; i < values[t].length; i++) {
				valuesSigned[t][i] = new int[values[t][i].length];
				for (int j = 0; j < values[t][i].length; j++) {
					valuesSigned[t][i][j] = (int) Math.signum(values[t][i][j]);
				}
			}
		}
	}

	/**
	 *Builds a cross reference mapping which as keys is alternative identifiers
	 * that map to a subset of genes in htgenes
	 */
	public void buildxref(HashSet<String> htregs,
			HashMap<String, HashSet<String>> htxref, String szxrefval)
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
				HashSet<String> htfoundgenes = new HashSet<String>(3);
				// htallsyn has all genes on the line
				HashSet<String> htallsyn = new HashSet<String>();
				// entries can be delimited by a tab, semicolon, pipe, comma, or
				// double quote
				StringTokenizer st = new StringTokenizer(szLine, "\"\t;|,");
				while (st.hasMoreTokens()) {
					String sztoken = st.nextToken().trim().toUpperCase();

					// adding to synonym list
					htallsyn.add(sztoken);

					if (htregs.contains(sztoken)) {
						// storing known genes
						htfoundgenes.add(sztoken);
					}
				}
				if (!htfoundgenes.isEmpty()) {
					// foundgenes not empty implies htallsynitr not empty
					Iterator<String> htallsynitr = htallsyn.iterator();
					while (htallsynitr.hasNext()) {
						String szsyn = htallsynitr.next();
						HashSet<String> hssyngenes = htxref.get(szsyn);

						if (hssyngenes == null) {
							hssyngenes = new HashSet<String>();
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
        //FIXME Modify the TF-DNA/ miRNA-target interaction file
        
	public void adjustBindingData(HashMap<String, Integer> reg2DataSetIndex,
			DREM_DataSet theDataSet, String szExFile,
                        DataSetCore miRNADataSet,String miRNAExFile,
                        DataSetCore proteDataSet,String djProteFile,String djPPIFile,Boolean bProteAll,
			boolean scaleMIRNAExp, boolean scaleTFExp,
			double miRNAScalingFactor, double minTFExp,
                        HashMap<String,HashMap<String,Double>> methyGeneScoreMap,String pweight) {
                
                    // Update adjustBinding
                    double[][]regTR; //first dimension: time, second dimension: Regulator 
                    double [][]proteTR; //regulator matrix from proteomics data
                    
                    //proteomics relative proteimics data weight
                    double pweightd=Double.parseDouble(pweight);
                   
                    
                    //update MethyG2R; Integrate methylation
                    if (methyGeneScoreMap!=null){
                        updateMethyG2R(methyGeneScoreMap,szExFile,theDataSet); //update Gene2RegBinding/Index using methylation data
                    }
                   
                    regTR=buildregTR(theDataSet,szExFile,miRNADataSet,reg2DataSetIndex);
                   
                    // Integrate proteomics data
                    proteTR=prote2Bind(theDataSet,szExFile,miRNADataSet,proteDataSet,djProteFile,djPPIFile,reg2DataSetIndex,bProteAll); //integrates proteomics dataset
                    
                    regTR=mergeTR(regTR,proteTR,pweightd); //integrates proteomics dataset
                    // Convert the regTR (regulator matrix) to gene2RegBinding matrix used by the program 
                    if (reg2DataSetIndex.size()>0){
                        for (int time=0;time<(theDataSet.numcols)-1;time++){
                            for (int geneIndex=0;geneIndex<gene2RegBinding[time].length;geneIndex++){
                                for (int numRegulator=0;numRegulator<gene2RegBindingIndex[time][geneIndex].length;numRegulator++){

                                    double bindingValue = gene2RegBinding[time][geneIndex][numRegulator];
                                    int bindingIndexValue = gene2RegBindingIndex[time][geneIndex][numRegulator];
                                    double regValue=regTR[time][bindingIndexValue];
                                    double geneValue=theDataSet.data[geneIndex][time+1]-theDataSet.data[geneIndex][time];
                                    int regDirection;

                                    if(regValue*geneValue<0){
                                        regDirection=-1;
                                    }else{
                                        regDirection=1;
                                        if (regTypes[bindingIndexValue]==RegulatorBindingData.MIRNA){
                                            regDirection=0;
                                        }
                                    }

                                    if (bindingValue==0){
                                        gene2RegBinding[time][geneIndex][numRegulator]=0;
                                    }else{
                                        if(regValue<minTFExp){
                                            gene2RegBinding[time][geneIndex][numRegulator]=regDirection*minTFExp*Math.abs(bindingValue);
                                        }else{
                                            gene2RegBinding[time][geneIndex][numRegulator]=regDirection*regValue*Math.abs(bindingValue);
                                        }
                                    }

                                    if (regPriors != null)
                                                        gene2RegBinding[time][geneIndex][numRegulator] *= regPriors[numRegulator];
                                }
                            }
                        }
                        
                        for (int time=0;time<theDataSet.numcols;time++){
                           makeRegulatorIndex(gene2RegBinding[time],
                                                gene2RegBindingIndex[time], reg2GeneBinding[time],
                                                reg2GeneBindingIndex[time]); 
                        }
                    }
                
                /*
		for (int time = 0; time < theDataSet.numcols; time++) { 
			for (int geneIndex = 0; geneIndex < gene2RegBinding[time].length; geneIndex++) {
				for (int numRegulator = 0; numRegulator < gene2RegBindingIndex[time][geneIndex].length; numRegulator++) {
                                    
					double bindingValue = gene2RegBinding[time][geneIndex][numRegulator];
					int bindingIndexValue = gene2RegBindingIndex[time][geneIndex][numRegulator];
                                        String regName=regNames[bindingIndexValue].toUpperCase();
                                        //
					Integer dataSetIndex = reg2DataSetIndex
							.get(regNames[bindingIndexValue].toUpperCase());
                                        
					if (bindingValue == 0) {
						gene2RegBinding[time][geneIndex][numRegulator] = 0;
					} else if (regTypes[bindingIndexValue] == RegulatorBindingData.MIRNA
							&& dataSetIndex != null
							&& scaleMIRNAExp
							&& time < timepoints - 1) {
						double value = miRNADataSet.data[dataSetIndex][time+1];
						if(Math.abs(value) < 0.25 ){
							gene2RegBinding[time][geneIndex][numRegulator] = 0;				
						}else{
							//System.out.println("Using scaling factor: "+miRNAScalingFactor);
							gene2RegBinding[time][geneIndex][numRegulator]=(int)Math.signum( logitShiftZeroNegative(value,miRNAScalingFactor));
							//gene2RegBinding[time][geneIndex][numRegulator]= logitShiftZeroNegative(value,miRNAScalingFactor);
						}
						
					} else if (regTypes[bindingIndexValue] == RegulatorBindingData.TF
							&& dataSetIndex != null
							&& scaleTFExp
							&& time < timepoints - 1) {
						double interValue = logitShiftZeroNegative(
								theDataSet.data[dataSetIndex][time + 1]
										- theDataSet.data[dataSetIndex][time],
								miRNAScalingFactor);
                                      
						if (Math.abs(interValue) < minTFExp)
							interValue = minTFExp * Math.signum(interValue);
                                                
                                                //gene2RegBinding[time][geneIndex][numRegulator] = interValue; //interValue;
						gene2RegBinding[time][geneIndex][numRegulator] = 1; //interValue;
                                                
					} else {
						gene2RegBinding[time][geneIndex][numRegulator] = bindingValue;
					}

					
					  //Scale the activity values by the regulator priors.
					 
					if (regPriors != null)
						gene2RegBinding[time][geneIndex][numRegulator] *= regPriors[numRegulator];
				}
			}
			makeRegulatorIndex(gene2RegBinding[time],
					gene2RegBindingIndex[time], reg2GeneBinding[time],
					reg2GeneBindingIndex[time]);
		}
               */
                
	}
       
       
        //FIXME: get time points for each dataset.
        //This was mainly used to handle the case with different time points for each type
        // of the data (expression, miRNA, proteomics,etc)
        public String[] getTimePoints(String FileName, Boolean badd0){
            String[] TimePoints=null;
            ArrayList<String> FTimePoints=new ArrayList<String>();
            try{
                    BufferedReader br=new BufferedReader(new FileReader(FileName));
                    String FR=br.readLine();
                    TimePoints=FR.split("\\\t");
                    TimePoints=Arrays.copyOfRange(TimePoints,1,TimePoints.length);
                    
                    if (badd0){
                        FTimePoints.add("Added_0");
                    }
                    for (String elem : TimePoints){
                        FTimePoints.add(elem);
                    }
                    TimePoints=FTimePoints.toArray(new String[FTimePoints.size()]);
            }catch (IOException e){
                e.printStackTrace();
            }
            return TimePoints;
        }
        
        //Update MethyTR (update TFDNA using methylation data
        
        public void updateMethyG2R(HashMap<String,HashMap<String,Double>> methyGeneScoreMap,String szExFile,DREM_DataSet theDataSet){
            
            //update implementation
            String[] GeneTimePoints=getTimePoints(szExFile,theDataSet.badd0);
            String regName;
            String timeName;
            String geneName;
            String key;
            double mScore; 
            double tfActivityFromMethy;
            
            for (int time=0;time<gene2RegBinding.length;time++){
                for (int geneIndex=0;geneIndex<gene2RegBinding[time].length;geneIndex++){
                    timeName=GeneTimePoints[time];
                    geneName=theDataSet.genenames[geneIndex].toUpperCase();
                    
                    if (methyGeneScoreMap.containsKey(timeName)){
                        if (methyGeneScoreMap.get(timeName).containsKey(geneName)){
                            mScore=methyGeneScoreMap.get(timeName).get(geneName);
                            tfActivityFromMethy=1-mScore;
                            for (int regIndex=0;regIndex<gene2RegBinding[time][geneIndex].length;regIndex++){
                                gene2RegBinding[time][geneIndex][regIndex]=gene2RegBinding[time][geneIndex][regIndex]*tfActivityFromMethy;
                            }
                        } 
                    } else{ 
                        if (time>0){
                            for (int regIndex=0;regIndex<gene2RegBinding[time][geneIndex].length;regIndex++){
                                gene2RegBinding[time][geneIndex][regIndex]=gene2RegBinding[time-1][geneIndex][regIndex];
                           }
                        }
                    }
                }
            }
     

        }
        
        
        //FIXME: build regTR based on expression  
        public double[][] buildregTR(DREM_DataSet theDataSet,String szExFile,DataSetCore miRNADataSet, HashMap<String,Integer> reg2DataSetIndex){
             // get regulator expression level 
            double [][] regTR=new double[theDataSet.numcols][regNames.length];
            double RegValue;
            for (int time=0;time<(theDataSet.numcols);time++){
                        for (int reg=0;reg<regNames.length;reg++){
                            RegValue=Double.NEGATIVE_INFINITY;// default value;
                            if (time<theDataSet.numcols-1){
                                String regName=regNames[reg].toUpperCase();
                                int regIndex;
                                if(regTypes[reg]==RegulatorBindingData.TF){
                                    regIndex=Arrays.asList(theDataSet.genenames).indexOf(regName);
                                    if (regIndex!=-1){
                                         RegValue = (theDataSet.data[regIndex][time+1]) -(theDataSet.data[regIndex][time]);
                                    }
                                }else if (regTypes[reg]==RegulatorBindingData.MIRNA){
                                    regIndex=Arrays.asList(miRNADataSet.genenames).indexOf(regName);
                                    if (regIndex!=-1){
                                         RegValue = (miRNADataSet.data[regIndex][time+1])-(miRNADataSet.data[regIndex][time]);
                                    }
                                }
                            }
                            regTR[time][reg]=RegValue;
                        }           
            }
            
            regTR=normalizeRegulatorLevel(regTR);
            return regTR;
        }
        
        //FIXME: integreate proteomics to TF binding 
        
        public double[][] prote2Bind(DREM_DataSet theDataSet,String szExFile,DataSetCore miRNADataSet, DataSetCore proteDataSet,String djProteFile,
                                    String djPPIFile,HashMap<String,Integer> reg2DataSetIndex, Boolean bProtAll){
            //FIXME: create a new hashmap for proteomics data
            //input: proteomics data
            //output: normalized [-1,1] TF actitivity level for each TF at each time poinT TFActProt=[time][numRegulator]
            HashMap<String,Integer> reg2ProteDataSetIndex =new HashMap<> ();
            HashMap<String,Double> ppi;
            HashMap<String,String> ppartner;
            ppi = new HashMap<>();
            ppartner=new HashMap<>();
            
            String[] GeneTimePoints=getTimePoints(szExFile,theDataSet.badd0);
            
            double[][] proteTR=new double[theDataSet.numcols][regNames.length];
            double protRegValue;
            
            for (int time=0;time<theDataSet.numcols;time++){
                        for (int reg=0;reg<regNames.length;reg++){
                            protRegValue=Double.NEGATIVE_INFINITY;// default value;
                        }
            }
            
            if (proteDataSet!=null
               && !"".equals(djPPIFile)){
                    // handle the time mapping
                    String[] ProteTimePoints=getTimePoints(djProteFile,proteDataSet.badd0);
                    int i;
                    int pi;
                    HashMap <Integer,Integer> TimeMap=new HashMap<>(); //Mapping time points between different data types
                    for (i=0;i<GeneTimePoints.length;i++){
                        pi=Arrays.asList(ProteTimePoints).indexOf(GeneTimePoints[i]);
                        if (pi!=-1){
                            TimeMap.put(i,pi);
                        }
                    }

                    HashMap <Integer,Integer> TimeMapP=new HashMap<>();
                    for (i=0;i<ProteTimePoints.length;i++){
                        pi=Arrays.asList(GeneTimePoints).indexOf(ProteTimePoints[i]);
                        if (pi!=-1){
                            TimeMapP.put(i,pi);
                        }
                    }

                    //
                    for (i=0;i<proteDataSet.genenames.length;i++){
                        reg2ProteDataSetIndex.put(proteDataSet.genenames[i],i);
                    }
                    String Line;
                    String[] LineSplit;
                    String UpdateVal;
                    try{
                        BufferedReader br = new BufferedReader(new FileReader(djPPIFile));
                        while ((Line=br.readLine())!=null){
                            Line=Line.trim();
                            LineSplit=Line.split("\\\t");
                            ppi.put(LineSplit[0].toUpperCase()+","+LineSplit[1].toUpperCase(),Double.parseDouble(LineSplit[2]));
                            if (!ppartner.containsKey(LineSplit[0])){
                                ppartner.put(LineSplit[0],LineSplit[1]);
                            }else{
                                UpdateVal=ppartner.get(LineSplit[0])+','+LineSplit[1];
                                ppartner.put(LineSplit[0],UpdateVal);
                            }
                        }
                    }catch (IOException e){
                        e.printStackTrace();  
                    }

                    //get the protein level for each regulator at each time point
                   
                    int ptime,ptimeplus;  //
                    
                    for (int time=0;time<theDataSet.numcols;time++){
                        for (int reg=0;reg<regNames.length;reg++){
                            protRegValue=Double.NEGATIVE_INFINITY;// default value;
                            String regName=regNames[reg].toUpperCase();
                            if (time<theDataSet.numcols-1){
                                int regIndex=Arrays.asList(proteDataSet.genenames).indexOf(regName);
                                if (regIndex!=-1){
                                    if (TimeMap.containsKey(time) && TimeMap.containsKey(time+1)){
                                         ptime=TimeMap.get(time);
                                         ptimeplus=TimeMap.get(time+1);
                                         protRegValue = (proteDataSet.data[regIndex][ptimeplus]) -(proteDataSet.data[regIndex][ptime]);
                                    }
                                }
                            }
                            proteTR[time][reg]=protRegValue;
                        }
                    }
                    
                    // normalize the protein level 
                    proteTR=normalizeProteinLevel(proteTR);
                    
                    // utilize ppi network 
                    if (bProtAll){
                        proteTR=integratePPI(theDataSet, proteDataSet,proteTR,ppi,ppartner,TimeMapP);
                        proteTR=normalizeProteinLevel(proteTR);
                    }
                    
                    
                    //merge regTR and proteTR
                   //regTR=mergeTR(regTR,proteTR);
                    
       
            }
            return proteTR;
        }
        
        //merge protTR and regTR
        public double[][] mergeTR(double[][] regTR,double[][]proteTR,double pweight){
             for (int time=0;time<regTR.length;time++){
                for (int reg=0;reg<regTR[time].length;reg++){
                    if (proteTR[time][reg]!=Double.NEGATIVE_INFINITY){
                        //regTR[time][reg]=proteTR[time][reg];
                        regTR[time][reg]=Math.pow(proteTR[time][reg],1.0/pweight);
                    }
                }
             }
             return regTR;
        }
        
        //integrate the PPI network 
       
        public double[][] integratePPI(DREM_DataSet theDataSet, DataSetCore proteDataSet, double[][] protTR, 
                                HashMap<String,Double> ppi, HashMap<String,String> ppartner,
                                HashMap<Integer,Integer> TimeMapP ){ 
    
            //Normalize the gene expression and protein data
            double [][] EX_Data=new double[theDataSet.genenames.length][theDataSet.numcols];
            for(int time=0;time<theDataSet.numcols;time++){
                for (int geneIndex=0;geneIndex<theDataSet.genenames.length;geneIndex++){
                     
                     if (time<theDataSet.numcols-1){
                        EX_Data[geneIndex][time]=theDataSet.data[geneIndex][time+1]-theDataSet.data[geneIndex][time];
                     }else{
                        EX_Data[geneIndex][time]=Double.NEGATIVE_INFINITY;
                     }
                }
            }
            EX_Data=normalizeList(EX_Data);
            
            //protein data
            double[][]  PR_Data=new double[proteDataSet.genenames.length][proteDataSet.numcols];
            for (int time=0; time<proteDataSet.numcols;time++){
                for (int geneIndex=0;geneIndex<proteDataSet.genenames.length;geneIndex++){
                    if (time<proteDataSet.numcols-1){
                        PR_Data[geneIndex][time]=proteDataSet.data[geneIndex][time+1]-proteDataSet.data[geneIndex][time];
                    }else{
                        PR_Data[geneIndex][time]=Double.NEGATIVE_INFINITY;
                    }
                }
            }
            PR_Data=normalizeList(PR_Data);
            
            //  Integrate the PPI and Protein data
            int regIndex,regIndexP;
            int regPIndex,regPIndexP;
            double regPVal;
            double regVal;
            String regName,regP,geneName,proteName;
            String Key;
            double PPIVal;
            ArrayList<Double> regPS=new ArrayList<>();
            double sumRegPList=0;
            int timeP;
            HashMap<String,Integer> mapRegIndex=new HashMap<>();
            HashMap<String,Integer> mapRegIndexP=new HashMap<>();
            
            // build Hashmap for gene index
            for (int geneIndex=0;geneIndex<theDataSet.genenames.length;geneIndex++){
                geneName=theDataSet.genenames[geneIndex];
                mapRegIndex.put(geneName,Arrays.asList(theDataSet.genenames).indexOf(geneName));   
            }
            
            // build Hashmap for prote index
            for (int proteIndex=0;proteIndex<proteDataSet.genenames.length;proteIndex++){
                proteName=proteDataSet.genenames[proteIndex];
                mapRegIndexP.put(proteName, Arrays.asList(proteDataSet.genenames).indexOf(proteName));
            }
            
           
             for (int time=0; time<proteDataSet.numcols;time++){
                timeP=TimeMapP.get(time);
                for (int reg=0; reg<regNames.length; reg++){
                    regPS.clear(); 
                    regName=regNames[reg].toUpperCase();
                    if (mapRegIndexP.containsKey(regName)){
                        regIndexP=mapRegIndexP.get(regName);
                        regVal=PR_Data[regIndexP][time];
                    }else if (mapRegIndex.containsKey(regName)){
                        regIndex=mapRegIndex.get(regName);
                        regVal=EX_Data[regIndex][timeP];
                    }else{
                        regVal=Double.NEGATIVE_INFINITY;
                        continue;
                    }
                    
                    for (int geneIndex=0;geneIndex<theDataSet.genenames.length;geneIndex++){
                        regP=theDataSet.genenames[geneIndex].toUpperCase();
                        if (mapRegIndexP.containsKey(regName)){
                            regPIndexP=mapRegIndexP.get(regName);
                            regPVal=PR_Data[regPIndexP][time];
                        }else if(mapRegIndex.containsKey(regName)){
                            regIndex=mapRegIndex.get(regName);
                            regPVal=EX_Data[regIndex][timeP];
                        }else{
                            regPVal=Double.NEGATIVE_INFINITY;
                            continue;
                        }
                        
                        regPVal=Math.abs(regPVal);
                        Key=regName+","+regP;
                        if (ppi.containsKey(Key) && regVal!=Double.NEGATIVE_INFINITY && regPVal!=Double.NEGATIVE_INFINITY){
                            PPIVal=ppi.get(Key);
                            regPS.add(regVal*PPIVal*regPVal);
                        }
                    }
                    
                    if (regPS.size()>0){
                        for (int i=0; i<regPS.size();i++){
                            sumRegPList+=regPS.get(i);
                        }
                        sumRegPList/=regPS.size();
                        protTR[time][reg]=sumRegPList;
                    }
                }
            }
            return protTR;
        }
        

        
        
        //Normalize all List 
        
        public double[][] normalizeList(double[][] DataList){
            ArrayList<Double> DList=new ArrayList<>();
            
            for (int i=0; i<DataList.length;i++){
                for (int j=0; j<DataList[i].length;j++){
                    if (DataList[i][j]!=Double.NEGATIVE_INFINITY){
                        DList.add(DataList[i][j]);
                    }
                }
            }
            
            Collections.sort(DList);
            
            double [] RankPoints={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
            double[] dataRank= new  double [11];
            int iindex;
            for (int i=0;i<RankPoints.length;i++){
                iindex=(int) (RankPoints[i] * (DList.size()-1));
                dataRank[i]=DList.get(iindex);
            }
            double dataw=estimateLogitW(RankPoints,dataRank);
            
            for (int i=0;i<DataList.length;i++){
                for (int j=0;j<DataList[i].length;j++){
                    if (DataList[i][j]!=Double.NEGATIVE_INFINITY){
                        DataList[i][j]=logitShiftZeroNegative(DataList[i][j],dataw);
                    }
                }
            }
            return DataList;
        }
        
        //normalize the regulator expression data
        public double[][] normalizeRegulatorLevel(double[][]regTR){
            
            ArrayList<Double> MirVal=new ArrayList<>();
            ArrayList<Double> TFVal=new ArrayList<>();
            for (int time=0;time<regTR.length;time++){
                for (int reg=0;reg<regTR[time].length;reg++){
                    int regType=regTypes[reg];
                    if(regType==RegulatorBindingData.TF && regTR[time][reg]!=Double.NEGATIVE_INFINITY && regTR[time][reg]!=Double.POSITIVE_INFINITY && !Double.isNaN(regTR[time][reg])){
                        TFVal.add(regTR[time][reg]);
                    }else if (regType==RegulatorBindingData.MIRNA && regTR[time][reg]!=Double.NEGATIVE_INFINITY && regTR[time][reg]!=Double.POSITIVE_INFINITY && !Double.isNaN(regTR[time][reg])){ 
                        MirVal.add(regTR[time][reg]);
                    }
                }
            }
            Collections.sort(TFVal);
            Collections.sort(MirVal);
            double [] TFRank=new double [11]; // top 0, top 10%, 20%, 30%,40%,50%,..., 100% 
            double [] MIRank=new double [11];
            double [] RankPoints={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
            
            double TFw=Double.NEGATIVE_INFINITY;
            double MIRw=Double.NEGATIVE_INFINITY;
            
            //TFRank
            if (TFVal.size()>0){
                for (int i=0;i<RankPoints.length;i++){
                    TFRank[i]=TFVal.get((int)(RankPoints[i]*(TFVal.size()-1)));
                }

                TFw=estimateLogitW(RankPoints,TFRank);
            }
            //miRNA rank
            if (MirVal.size()>0){
                for (int i=0;i<RankPoints.length;i++){
                    MIRank[i]=MirVal.get((int)(RankPoints[i]*(MirVal.size()-1)));
                }
                MIRw=estimateLogitW(RankPoints,MIRank);
            }
            
           
            for(int time=0;time<regTR.length;time++){
                for (int reg=0;reg<regTR[time].length;reg++){
                    int regType=regTypes[reg];
                    if(regType==RegulatorBindingData.TF && regTR[time][reg]!=Double.NEGATIVE_INFINITY){
                        regTR[time][reg]=logitShiftZeroNegative(regTR[time][reg],TFw);
                    }else if (regType==RegulatorBindingData.MIRNA && regTR[time][reg]!=Double.NEGATIVE_INFINITY){ 
                        regTR[time][reg]=logitShiftZeroNegative(regTR[time][reg],MIRw);
                    }
                }
            }
            return regTR;
        }
        
        //normalize the proteomics data
        public double[][] normalizeProteinLevel(double[][] protTR){
            ArrayList<Double> ProteVal=new ArrayList<>();
            //get MaxProt and minProt
            for (int time=0;time<protTR.length;time++){
                for (int reg=0;reg<protTR[time].length;reg++){
                    if (protTR[time][reg]!=Double.NEGATIVE_INFINITY){
                        ProteVal.add(protTR[time][reg]);
                    }
                }
            }
            Collections.sort(ProteVal);
            //Normalize
            double [] RankPoints={0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0};
            double [] ProtRank=new double [11];
            double w;
            for (int i=0;i<RankPoints.length;i++){
                ProtRank[i]=ProteVal.get((int)(RankPoints[i]*(ProteVal.size()-1)));
            }
            
            w=estimateLogitW(RankPoints,ProtRank);
            for (int time=0;time<protTR.length;time++){
                for (int reg=0;reg<protTR[time].length;reg++){
                    if (protTR[time][reg]!=Double.NEGATIVE_INFINITY){
                        protTR[time][reg]=logitShiftZeroNegative(protTR[time][reg],w);
                    } 
                }
            }
            return protTR;
        }
        
        //integrate ppi network 
        //estimate the proper W based on protein level distribution
        
        public double estimateLogitW(double[] rankp, double[] rankv){
            double error=0,minError;
            double rankpi;
            minError=Double.POSITIVE_INFINITY;
            HashMap <Double,Double> VW=new HashMap<>();
            double v;
            for (double w=0.1;w<6;w=w+0.1){
                error=0;
                for (int i=0;i<rankp.length;i++){
                   v=logitShiftZeroNegative(rankv[i],w);
                   rankpi=(rankp[i]-0.5)*2;
                   error+=Math.abs(v-rankpi);
                }
                if (error<minError){
                    minError=error;
                }
                VW.put(error,w);
            }
            double chosenW=VW.get(minError);
            return chosenW;
        }
        
        //logistic function
        
	public double logit(double x, double w) {
		return (1 / (1 + Math.exp(-w * x)));
	}

	/**
	 * A modified version of the logit function that shifts the logit function
	 * to 0, s.t. when x is 0 the function returns 0, instead of 0.5 normally.
	 * Also instead of [0,1] the function outputs values in the range [-1,1] to
	 * allow for negative influence. As with a normal logit function the weight
	 * w can be used to control the steepness of the function. Currently, it is
	 * assumed that the function is only called for entries that are supposed to
	 * be regulating, that means if there is no regulatory relationship than
	 * logitShiftZeroNegative should always return 0, but that has to be
	 * controlled before.
	 */

	public double logitShiftZeroNegative(double x, double w) {
		if (x == 0) {
			return (0);
		} else {
			return (Math.signum(x) * ((logit(Math.abs(x), w) - 0.5) * 2));
		}
	}

	/**
	 * This class is responsible for manipulating the TF-gene data so that it
	 * will be in format readily usable by a weighted classifier. For a split
	 * with n children, the classifier needs n copies of each instance because
	 * each instance will be assumed to have every label, but with different
	 * weights for different labels.
	 */
	public void loadInstances(int nmaxchild, int numbits) {
		// first dimension determines the number of instances we will have
		// each second dimension corresponds to an instance
		// the third dimension is the values in the instances

		theInstances = new double[timepoints][nmaxchild - 1][][];
		ylabels = new int[timepoints][nmaxchild - 1][];
		theInstancesIndex = new int[timepoints][nmaxchild - 1][][];
		theInstancesReg = new double[timepoints][nmaxchild - 1][][];
		theInstancesRegIndex = new int[timepoints][nmaxchild - 1][][];
		for (int numchild = 2; numchild <= nmaxchild; numchild++) {
			for (int t = 0; t < timepoints; t++) {
				theInstancesIndex[t][numchild - 2] = new int[numchild
						* gene2RegBindingIndex[t].length][];
				theInstancesRegIndex[t][numchild - 2] = new int[numchild
						* numbits][];
				theInstances[t][numchild - 2] = new double[numchild
						* gene2RegBinding[t].length][];
				theInstancesReg[t][numchild - 2] = new double[numchild
						* numbits][];
				ylabels[t][numchild - 2] = new int[numchild
						* gene2RegBinding[t].length];
				int nspot = 0;
				for (int ngene = 0; ngene < gene2RegBinding[t].length; ngene++) {
					for (int nchild = 0; nchild < numchild; nchild++) {
						theInstances[t][numchild - 2][nspot] = gene2RegBinding[t][ngene];
						theInstancesIndex[t][numchild - 2][nspot] = gene2RegBindingIndex[t][ngene];
						ylabels[t][numchild - 2][nspot] = nchild;
						nspot++;
					}
				}

				makeRegulatorIndex(theInstances[t][numchild - 2],
						theInstancesIndex[t][numchild - 2],
						theInstancesReg[t][numchild - 2],
						theInstancesRegIndex[t][numchild - 2]);

			}
		}
	}

}


class gene2Reg{
    public double[][] gene2RegBinding;
    public int[][] gene2RegBindingIndex;
    public gene2Reg(double[][] gene2RegBindingV, int[][] gene2RegBindingIndexV){
        gene2RegBinding=gene2RegBindingV;
        gene2RegBindingIndex=gene2RegBindingIndexV;
    }
    public gene2Reg(){
    }
}
